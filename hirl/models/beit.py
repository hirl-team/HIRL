from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hirl import layers
from hirl.backbones.vision_transformer import *
from hirl.backbones.dvae import *


class BEiT(nn.Module):
    """
    Re-implementation of BERT Pre-Training of Image Transformers (BEiT)
    based on https://github.com/microsoft/unilm/tree/master/beit

    Args:
        vit_model (nn.Module): the Vision Transformer model
        dvae_model (nn.Module): the discrete VAE model
        num_mlp_layer (int): the number of MLP layers in projection head (default: 0)
        activation (str or function, optional): activation function for projection head (default: "relu")
    """

    def __init__(self, vit_model, dvae_model, num_mlp_layer=0, activation="relu", **kwargs):
        super().__init__()

        self.vit_model = vit_model
        self.dvae_model = dvae_model
        hidden_dim = vit_model.head.weight.shape[1]
        if num_mlp_layer > 0:
            self.projector = layers.MLP(hidden_dim, [hidden_dim] * num_mlp_layer,
                                        activation=activation, batch_norm=True)
        else:
            self.projector = nn.Identity()
        self.criterion = nn.CrossEntropyLoss()

        # create hooks for the embeddings before and after projection
        self.entangled_dim = hidden_dim
        self.inst_dim = hidden_dim
        self._embeddings = {}

        def hook_function(module, input_emb, output_emb):
            if isinstance(input_emb, tuple) and len(input_emb) == 1:
                input_emb = input_emb[0]
            if isinstance(output_emb, tuple) and len(output_emb) == 1:
                output_emb = output_emb[0]
            entangled_cls_emb = input_emb[:, 0]
            entangled_mean_emb = torch.mean(input_emb[:, 1:], dim=1)
            inst_cls_emb = output_emb[:, 0]
            inst_mean_emb = torch.mean(output_emb[:, 1:], dim=1)
            self._embeddings["entangled_cls_emb"] = entangled_cls_emb
            self._embeddings["entangled_mean_emb"] = entangled_mean_emb
            self._embeddings["inst_cls_emb"] = inst_cls_emb
            self._embeddings["inst_mean_emb"] = inst_mean_emb

        self.projector.register_forward_hook(hook_function)

    def get_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss

    def forward_feature(self, images):
        cls_emb = self.vit_model(images, return_all_tokens=False, use_head=False)
        proj_cls_emb = self.projector(cls_emb)
        return proj_cls_emb

    def forward(self, images, dvae_images, token_mask, return_hooks=False):
        # Get codebook indices and labels for masked image modeling
        token_mask = token_mask.to(torch.bool)
        with torch.no_grad():
            input_ids = self.dvae_model.get_codebook_indices(dvae_images).flatten(1)
            seq_token_mask = token_mask.flatten(1)
            target = input_ids[seq_token_mask]

        # Masked position prediction
        embedding_dict = dict()
        emb = self.vit_model(images, mask=token_mask, return_all_tokens=True)
        proj_emb = self.projector(emb)
        pred = self.vit_model.head(proj_emb[:, 1:][seq_token_mask])
        loss = self.get_loss(pred, target)

        output_dict = dict(loss=loss)
        if return_hooks:
            for k, v in self._embeddings.items():
                embedding_dict[k] = v
            output_dict["embeddings"] = embedding_dict

        return output_dict


def beit_base(**kwargs):
    vit_model = vit_base(patch_size=16, num_classes=8192, masked_im_modeling=True)
    dvae_model = DiscreteVAE(image_size=112)
    model = BEiT(vit_model, dvae_model, **kwargs)
    return model

def beit_base_with_proj(**kwargs):
    vit_model = vit_base(patch_size=16, num_classes=8192, masked_im_modeling=True)
    dvae_model = DiscreteVAE(image_size=112)
    model = BEiT(vit_model, dvae_model, num_mlp_layer=2, **kwargs)
    return model

def beit_large(**kwargs):
    vit_model = vit_large(patch_size=16, num_classes=8192, masked_im_modeling=True)
    dvae_model = DiscreteVAE(image_size=112)
    model = BEiT(vit_model, dvae_model, **kwargs)
    return model

def beit_large_with_proj(**kwargs):
    vit_model = vit_large(patch_size=16, num_classes=8192, masked_im_modeling=True)
    dvae_model = DiscreteVAE(image_size=112)
    model = BEiT(vit_model, dvae_model, num_mlp_layer=2, **kwargs)
    return model
