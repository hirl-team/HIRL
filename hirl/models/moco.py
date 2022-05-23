import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MomentumContrast(nn.Module):
    """
    Re-implementation of Momentum Contrast v2 (MoCo v2) 
    based on https://github.com/facebookresearch/moco.

    Args:
        base_arch (str): the base architecture of encoder (default: "resnet50")
        dim (int): feature dimension (default: 128)
        queue_length (int): size of the queue for negative sampling (default: 16384)
        m (float): momentum for updating key encoder (default: 0.999)
        T (float): temperature parameter for InfoNCE loss (default: 0.2)
        mlp (bool): whether to use mlp projection (default: True)
        multi_crop (bool): whether to use multi-crops augmentation (default: False)
    """
    def __init__(self, base_arch="resnet50", dim=128, queue_length=16384, m=0.999, T=0.2, 
                    mlp=True, multi_crop=False, **kwargs):
        super().__init__()

        self.queue_length = queue_length
        self.m = m
        self.T = T
        self.multi_crop = multi_crop

        # create encoders and projection heads
        self.encoder_q = models.__dict__[base_arch](num_classes=dim)
        self.encoder_k = models.__dict__[base_arch](num_classes=dim)

        self.backbone = self.encoder_q

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        if mlp:
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # initialize key encoder's parameters as those of query encoder; not update key encoder with gradients 
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False

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
        project_layer = dict([*self.encoder_q.named_modules()])["fc"]
        project_layer.register_forward_hook(hook_function)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, queue_length))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def concat_all_gather(self, x):
        x_gather = [torch.ones_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(x_gather, x, async_op=False)
        output = torch.cat(x_gather, dim=0)
        return output

    def momentum_update_key_encoder(self):
        self._momentum_update_key_encoder()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = self.concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_length % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_length 
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffle index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # restore index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    @torch.no_grad()
    def extract_key_feat(self, im_k):
        self._momentum_update_key_encoder()  
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  
        k = F.normalize(k, dim=1)
        # undo shuffle
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k
        
    def extract_feat(self, images, is_eval=False):
        # global views
        if is_eval:
            return F.normalize(self.encoder_q(images), dim=1)
 
        im_q, im_k = images[0], images[1]  
        q = self.encoder_q(im_q)  # queries: NxC

        embedding_dict = dict()
        for k, v in self._embeddings.items():
            embedding_dict[k] = v

        q = nn.functional.normalize(q, dim=1)        
        # compute key features
        if self.multi_crop:
            k = self.extract_key_feat(im_k)
            local_views = list()
            for n, im_local in enumerate(images[2:]):
                local_q = self.encoder_q(im_local)
                local_q = nn.functional.normalize(local_q, dim=1)
                local_views.append(local_q)

            return q, k, local_views, embedding_dict
        else:
            k = self.extract_key_feat(im_k)
            # compute query features
            return q, k, None, embedding_dict

    def compute_local_logits(self, k, local_views):
        l_pos_list = [torch.einsum('nc,nc->n', [local_view, k]).unsqueeze(-1) for local_view in local_views]
        l_neg_list = [torch.einsum('nc,ck->nk', [local_view, self.queue.clone().detach()]) for local_view in local_views]
        local_logits = [torch.cat([l_pos, l_neg], dim=1) / self.T for (l_pos, l_neg) in zip(l_pos_list, l_neg_list)]
        local_labels = [torch.zeros(logit.shape[0], dtype=torch.long, device=logit.device) for logit in local_logits]
        return local_logits, local_labels

    def get_loss(self, logits, targets, local_logits, local_targets):
        loss = torch.tensor(0, dtype=torch.float32, device=logits.device)
        loss_global = F.cross_entropy(logits, targets)
        loss += loss_global

        if local_logits is not None:
            loss_local = torch.tensor(0, dtype=torch.float32, device=logits.device)
            for logit, target in zip(local_logits, local_targets):
                loss_local += F.cross_entropy(logit, target)
            loss += loss_local
        return loss

    def forward_feature(self, images):
        return self.encoder_k(images)

    def forward(self, images, return_hooks=False):
        """
        Given a list of images, compute cross-view contrastive loss with a 
        negative sample queue.

        Args:
            images (list of torch.Tensor [N,C,H,W]): image tensors. The first two 
                elements are global views.
            return_hooks (bool): whether return the hidden output by hook function. (default: False)

        Returns:
            output_dict: dict with the following fields:
                ``loss`` (torch.Tensor [1,]): loss for backward.
                ``logits`` (torch.Tensor [N, queue_length+1]): computed contrastive logits between global views.
                ``local_logits`` (list of torch.Tensor [N, queue_length+1]): computed contrastive logits between 
                    local views and global views.
                ``targets`` (torch.Tensor [N, ]): indicating the positive pair index between global views.
                ``local_targets`` (list of torch.Tensor [N,]) indicating the positive pair index between local views and global views.
                ``embeddings``: dict used for further loss computation in hirl.
        """
        ## if images is not a list but a tensor, just extract feature
        if isinstance(images, torch.Tensor):
            images = [images]
        if len(images) == 1:
            image_emb = self.extract_feat(images, is_eval=True)
            if return_hooks:
                return image_emb, self._embeddings
            else:
                return image_emb

        q, k, local_views, embedding_dict = self.extract_feat(images)
        image_emb = [q, k] + local_views if self.multi_crop else [q, k]

        # derive logits and targets for contrastive learning
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        local_logits, local_targets = None, None
        if self.multi_crop:
            local_logits, local_targets = self.compute_local_logits(k, local_views)
        self._dequeue_and_enqueue(k)

        loss = self.get_loss(logits, targets, local_logits, local_targets)

        output_dict = dict(loss=loss,
                      logits=logits,
                      local_logits=local_logits,
                      targets=targets,
                      local_targets=local_targets,
                      )
        if return_hooks:
            output_dict["embeddings"] = embedding_dict
        
        return output_dict


def moco_resnet50_single_crop(**kwargs):
    model = MomentumContrast(base_arch="resnet50", multi_crop=False, **kwargs)
    return model

def moco_resnet50_multi_crop(**kwargs):
    model = MomentumContrast(base_arch="resnet50", multi_crop=True, **kwargs)
    return model

def moco_resnet50(**kwargs):
    model = MomentumContrast(base_arch="resnet50", **kwargs)
    return model

# set recommended models
moco_single = moco_resnet50_single_crop
moco_multi = moco_resnet50_multi_crop
