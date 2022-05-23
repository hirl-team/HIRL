from collections import defaultdict

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from hirl.utils import dist


class SwAV(nn.Module):
    """
    Re-implementation of Swapping Assignments between Views (SwAV)
    based on https://github.com/facebookresearch/swav.

    Args:
        base arch (str): the base architecture of encoder (default: "resnet50")
        dim (int): feature dimension (default: 128)
        batch_size (int): the number of samples on each GPU (default: 64)
        queue_length (int): size of the queue to supplement mini-batch samples (default: 0)
        T (float): temperature parameter for cluster assignment prediction (default: 0.1)
        epsilon (float): regularization parameter for Sinkhorn-Knopp algorithm (default: 0.05)
        mlp (bool): whether to use mlp projection (default: True)
        num_proto (int): number of prototypes (default: 3000)
        num_sinkhorn (int): number of iterations for Sinkhorn-Knopp algorithm (default: 3).
        multi_crop (bool): whether to use multi-crop augmentation (default: False)
        normalize (bool): whether to apply l2 normalization to output embedding (default: True)
    """
    def __init__(self, base_arch="resnet50", dim=128, batch_size=64, queue_length=0, T=0.1, epsilon=0.05, mlp=True, 
                    hidden_mlp=2048, num_proto=3000, num_sinkhorn=3, multi_crop=False, normalize=True, **kwargs):
        super().__init__()

        self.world_size = dist.get_world_size()
        self.batch_size = batch_size
        self.queue_length = queue_length
        self.T = T
        self.epsilon = epsilon
        self.num_proto = num_proto
        self.num_sinkhorn = num_sinkhorn
        self.multi_crop = multi_crop
        self.normalize = normalize

        # create encoder and projection head
        self.encoder = models.__dict__[base_arch](num_classes=dim)
        self.backbone = self.encoder
        dim_mlp = self.encoder.fc.weight.shape[1]
        if mlp:
            ## with a batch norm
            self.encoder.fc = nn.Sequential(
                nn.Linear(dim_mlp, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, dim)
            )

        # create prototypes and queue
        self.prototypes = nn.Linear(dim, num_proto, bias=False)
        self.queue_length -= self.queue_length % (self.world_size * self.batch_size)
        self.use_queue = self.queue_length >= self.world_size
        if self.use_queue:
            self.register_buffer("queue", torch.zeros(2, self.queue_length // self.world_size, dim))

        # initialize model weights
        self.apply(self._init_weights)

        # create hooks for the embeddings before and after projection
        self.entangled_dim = dim_mlp
        self.inst_dim = dim
        self._embeddings = {}
        def hook_function(module, input_emb, output_emb):
            if isinstance(input_emb, tuple) and len(input_emb) == 1:
                input_emb = input_emb[0]
            if isinstance(output_emb, tuple) and len(output_emb) == 1:
                output_emb = output_emb[0]
            self._embeddings["entangled_emb"] = input_emb
            self._embeddings["inst_emb"] = output_emb
        project_layer = dict([*self.encoder.named_modules()])["fc"]
        project_layer.register_forward_hook(hook_function)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def extract_feat(self, images):
        # extract feature of global views
        im_q, im_k = images[0], images[1]
        q = self.encoder(im_q)
        embedding_dict = defaultdict(list)
        for k,v in self._embeddings.items():
            embedding_dict[k].append(v)

        k = self.encoder(im_k)
        if self.normalize:
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)

        # extract features of local views
        if self.multi_crop:
            local_views = []
            for im_local in images[2:]:
                local_q = self.encoder(im_local)
                if self.normalize:
                    local_q = F.normalize(local_q, dim=1)
                local_views.append(local_q)
                for key, v in self._embeddings.items():
                    embedding_dict[key].append(v)
            return q, k, local_views, embedding_dict
        else:
            return q, k, None, embedding_dict

    @torch.no_grad()
    def distributed_sinkhorn(self, logit):
        """
        Clustering assignment by iterative Sinkhorn-Knopp algorithm (designed for distributed training).
        """
        Q = torch.exp(logit / self.epsilon).t() # [D, N]
        B = Q.shape[1] * self.world_size
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        torch_dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.num_sinkhorn):
            sum_row = torch.sum(Q, dim=1, keepdim=True)
            torch_dist.all_reduce(sum_row)
            Q /= sum_row
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def get_loss(self, q, k, logit_q, logit_k, logit_local):
        loss = torch.tensor(0, dtype=torch.float32, device=q.device)
        _tgt_logits = []
        _src_logits = []
        crops = [(q, logit_q), (k, logit_k)]

        for crop_id, (tgt_crop_id, src_crop_id) in enumerate([(0, 1), (1, 0)]):
            tgt_emb, tgt_logit = crops[tgt_crop_id]
            src_emb, src_logit = crops[src_crop_id]

            # get prediction target
            with torch.no_grad():
                tgt_emb = tgt_emb.detach()
                tgt_logit = tgt_logit.detach()
                if self.use_queue:
                    if not torch.all(self.queue[crop_id, -1, :] == 0):
                        tgt_logit = torch.cat([self.prototypes(self.queue[crop_id]), tgt_logit], dim=0)
                    self.queue[crop_id, self.batch_size:] = self.queue[crop_id, :-self.batch_size].clone()
                    self.queue[crop_id, :self.batch_size] = tgt_emb
                # Sinkhorn iteration
                tgt_assignment = self.distributed_sinkhorn(tgt_logit)[-self.batch_size:]

            # loss of clustering assignment prediction
            all_src_logit = torch.cat([src_logit] + logit_local, dim=0)  # ((N_local + 1) * B, K)
            all_src_logit /= self.T
            all_q = torch.cat([tgt_assignment] * (len(logit_local) + 1), dim=0)  # ((N_local + 1) * B, K)
            loss_ = -torch.mean(torch.sum(all_q * F.log_softmax(all_src_logit, dim=1), dim=1))
            loss += loss_
            _tgt_logits.append(all_q)
            _src_logits.append(all_src_logit)

        loss /= 2 ## average over crops for assignment
        _tgt_logit = torch.cat(_tgt_logits, dim=0)
        _src_logit = torch.cat(_src_logits, dim=0)
        return loss, _tgt_logit, _src_logit

    def forward_feature(self, images):
        return self.encoder(images)

    def forward(self, images, return_hooks=False):
        """
        Given a list of images, compute cross-view cluster assignment prediction loss with sinkhorn 
        assignment algorithm.

        Args:
            images (list of torch.Tensor [N,C,H,W]): image tensors. The first two 
                elements are global views.
            return_hooks (bool): whether return the hidden output by hook function. (default: False)

        Returns:
            output_dict: dict with the following fields:
                ``loss`` (torch.Tensor [1,]): loss for backward.
                ``target_logits`` (list of torch.Tensor [N,C]) cluster assignment logits for 
                    target views.
                ``source_logits`` (list of torch.Tensor [N,C]) cluster assignment logits for 
                    source views.
                ``embeddings``: dict used for further loss computation in hirl.
        """
        # normalize prototypes
        if self.normalize:
            with torch.no_grad():
                weight = self.prototypes.weight.data.clone()
                weight = F.normalize(weight, dim=1, p=2)
                self.prototypes.weight.copy_(weight)

        # feature extraction
        q, k, local_views, embedding_dict = self.extract_feat(images)
        image_emb = [q, k] + local_views if self.multi_crop else [q, k]
        logit_q = self.prototypes(q)
        logit_k = self.prototypes(k)
        logit_local = [self.prototypes(local_view) for local_view in local_views] if self.multi_crop else []

        loss, tgt_logit, src_logit = self.get_loss(q, k, logit_q, logit_k, logit_local)

        output_dict = dict(loss=loss,
                           target_logits=tgt_logit,
                           source_logits=src_logit)
                    
        if return_hooks:
            output_dict["embeddings"] = embedding_dict
 
        return output_dict


def swav_resnet50_single_crop(batch_size=64, queue_length=0, **kwargs):
    model = SwAV(base_arch="resnet50", batch_size=batch_size, queue_length=queue_length, 
                    multi_crop=False, **kwargs)
    return model

def swav_resnet50_multi_crop(batch_size=64, queue_length=0, **kwargs):
    model = SwAV(base_arch="resnet50", batch_size=batch_size, queue_length=queue_length, 
                    multi_crop=True, **kwargs)
    return model

# set recommended models
swav_single = swav_resnet50_single_crop
swav_multi = swav_resnet50_multi_crop
