from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hirl.backbones.vision_transformer import *


class MomentumContrastV3(nn.Module):
    """
    Re-implementation of Momentum Contrast v3 (MoCo v3) 
    based on hhttps://github.com/facebookresearch/moco-v3

    Args:
        encoder_q (nn.Module): the base encoder for queries
        encoder_k (nn.Module): the momentum encoder for keys
        dim (int): encoder output dimension (default: 256)
        mlp_dim (int): hidden dimension in MLP (default: 4096)
        T (float): temperature parameter for InfoNCE loss (default: 0.2)
        multi_crop (bool): whether to use multi-crops augmentation (default: False)
    """
    def __init__(self, encoder_q, encoder_k, dim=256, mlp_dim=4096, T=0.2, 
                    multi_crop=False, **kwargs):
        super().__init__()

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.backbone = self.encoder_q
        self.T = T
        self.multi_crop = multi_crop

        # projectors
        hidden_dim = self.encoder_q.head.weight.shape[1]
        del self.encoder_q.head, self.encoder_k.head 
        self.encoder_q.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.encoder_k.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

        # initialize key encoder's parameters as those of query encoder; not update key encoder with gradients 
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False

        # create hooks for the embeddings before and after projection
        self.entangled_dim = hidden_dim
        self.inst_dim = dim
        self._embeddings = {}
        def hook_function(module, input_emb, output_emb):
            # always unpack the input and output first
            if isinstance(input_emb, tuple) and len(input_emb) == 1:
                input_emb = input_emb[0]
            if isinstance(output_emb, tuple) and len(output_emb) == 1:
                output_emb = output_emb[0]
            self._embeddings["entangled_cls_emb"] = input_emb
            self._embeddings["inst_cls_emb"] = output_emb
        project_layer = dict([*self.encoder_q.named_modules()])["head"]
        project_layer.register_forward_hook(hook_function)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def concat_all_gather(self, x):
        """
        Performs `all_gather` operation on the provided tensors.
        Warning: torch.distributed.all_gather has no gradient.
        """
        x_gather = [torch.ones_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(x_gather, x, async_op=False)
        output = torch.cat(x_gather, dim=0)
        return output

    def momentum_update_key_encoder(self):
        self._momentum_update_key_encoder()

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def extract_key_feat(self, im_k):
        """
        Feature extraction of key image.
        """
        k = self.encoder_k(im_k)  
        k = F.normalize(k, dim=1)   
        return k
        
    def extract_feat(self, images, m, is_eval=False):
        # global views
        if is_eval:
            return F.normalize(self.encoder_q(images), dim=1)
 
        im_q, im_k = images[0], images[1]  
        ## get query
        q1 = self.predictor(self.encoder_q(im_q))  # queries: NxC
        embedding_dict = defaultdict(list)
        for key, val in self._embeddings.items():
            embedding_dict[key].append(val)
        q2 = self.predictor(self.encoder_q(im_k))

        q1 = F.normalize(q1, dim=1)
        q2 = F.normalize(q2, dim=1)   
        q = [q1, q2]
        ## get key
        with torch.no_grad():
            self._momentum_update_key_encoder(m) # update once 
            k1, k2 = self.extract_key_feat(im_q), self.extract_key_feat(im_k)
            k = [k2, k1]
        ## get local views
        local_views = None
        if self.multi_crop:
            local_views = list()
            for n, im_local in enumerate(images[2:]):
                local_q = self.predictor(self.encoder_q(im_local))
                for key, val in self._embeddings.items():
                    embedding_dict[key].append(val)
                local_q = F.normalize(local_q, dim=1)
                local_views.append(local_q)
        return q, k, local_views, embedding_dict

    def compute_local_logits(self, k, local_views):
        """
        Compute the logits and targets for local views.
        """
        local_logits = [torch.einsum('nc,mc->nm', [local_view, k]) / self.T for local_view in local_views]
        local_targets = []
        for logits in local_logits:
            N = logits.shape[0]  
            local_targets.append(torch.arange(N, dtype=torch.long, device=logits.device) + N * torch.distributed.get_rank())
        
        return local_logits, local_targets

    def get_loss(self, logits, targets, local_logits, local_targets):
        loss = torch.tensor(0, dtype=torch.float32, device=logits.device)
        loss_global = F.cross_entropy(logits, targets) * (2 * self.T)
        loss += loss_global

        if local_logits is not None:
            loss_local = torch.tensor(0, dtype=torch.float32, device=logits.device)
            for logit, target in zip(local_logits, local_targets):
                loss_local += F.cross_entropy(logit, target) * (2 * self.T)
                loss_local /= len(local_logits)
            loss += loss_local
        return loss

    def forward_feature(self, images):
        return self.encoder_k(images)

    def forward(self, images, m, return_hooks=False):
        """
        Given a list of images, compute cross-view contrastive loss within mini batch.

        Args:
            images (list of torch.Tensor [N,C,H,W]): image tensors. The first two 
                elements are global views.
            m (float): momentum for updating the momentum encoder.
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
            image_emb = self.extract_feat(images[0], is_eval=True)
            if return_hooks:
                return image_emb, self._embeddings
            else:
                return image_emb

        queries, keys, local_views, embedding_dict = self.extract_feat(images, m)
        ## Note: q is a list, k is a list
        image_emb = queries + keys + local_views if self.multi_crop else (queries+keys)

        losses = torch.tensor(0, dtype=torch.float32, device=images[0].device)
        logits = []
        targets = []
        local_logits = []
        local_targets = []

        for q_id, q in enumerate(queries):
            # derive logits and targets for contrastive learning
            k = keys[q_id] # already cross viewd in extract_feat()
            k = self.concat_all_gather(k)
            logit = torch.einsum('nc,mc->nm', [q, k]) / self.T
            N = logit.shape[0]  
            target = torch.arange(N, dtype=torch.long, device=logit.device) + N * torch.distributed.get_rank()
            local_logit, local_target = None, None
            if self.multi_crop:
                local_logit, local_target = self.compute_local_logits(k, local_views)

            loss = self.get_loss(logit, target, local_logit, local_target)
            losses += loss
            logits.append(logit)
            targets.append(target)
            local_logits.append(local_logit)
            local_targets.append(local_target)

        output_dict = dict(loss=losses,
                      logits=logits,
                      local_logits=local_logits,
                      targets=targets,
                      local_targets=local_targets,
                      )
        if return_hooks:
            output_dict["embeddings"] = embedding_dict
        
        return output_dict


def mocov3_vit_small_single_crop(**kwargs):
    encoder_q = vit_small(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_small(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=False, **kwargs)
    return model

def mocov3_vit_small_multi_crop(**kwargs):
    encoder_q = vit_small(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_small(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=True, **kwargs)
    return model

def mocov3_vit_base_single_crop(**kwargs):
    encoder_q = vit_base(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_base(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=False, **kwargs)
    return model

def mocov3_vit_base_multi_crop(**kwargs):
    encoder_q = vit_base(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_base(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=True, **kwargs)
    return model

def mocov3_vit_large_single_crop(**kwargs):
    encoder_q = vit_large(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_large(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=False, **kwargs)
    return model

def mocov3_vit_large_multi_crop(**kwargs):
    encoder_q = vit_large(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    encoder_k = vit_large(patch_size=16, num_classes=4096, sincos_pos_emb=True, mocov3_init=True, use_head=True)
    model = MomentumContrastV3(encoder_q, encoder_k, multi_crop=True, **kwargs)
    return model
