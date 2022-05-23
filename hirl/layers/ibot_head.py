import torch.nn as nn

from .dino_head import DINOHead


class iBOTHead(DINOHead):
    """
    Head used in iBOT to map latent representation into prototype assignment.
    Reference: https://github.com/bytedance/ibot. Inherent from DINO Head.

    Args:
        patch_out_dim (int): output dimension of patch embedding. (default: 8192)
        norm (str): norm layer used in head. If None, use no norm layer. (default: None)
        act (str): activation layer used in MLP. (default: gelu)
        last_norm (str): norm layer used after last layer. If None, no such a norm layer. (default: None)
        nlayers (int): number of hidden layers in MLP. (default: 3)
        bottleneck_dim (int): hidden dimension in MLP. (default: 256)
        norm_last_layer (bool): whether to normlaize the last layer output. (default: False)
        shared_head (bool): for patch output, whether to use the same head as cls token output. (default: False)
    """
    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=False, 
                 shared_head=True, **kwargs):
        
        super(iBOTHead, self).__init__(*args,
                                        norm=norm,
                                        act=act,
                                        last_norm=last_norm,
                                        nlayers=nlayers,
                                        hidden_dim=hidden_dim,
                                        bottleneck_dim=bottleneck_dim,
                                        norm_last_layer=norm_last_layer, 
                                        **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        
        return x1, x2
