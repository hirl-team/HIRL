from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hirl import layers
from hirl.backbones.vision_transformer import *
from torch.nn.parallel.distributed import DistributedDataParallel


class DINO(nn.Module):
    """
    Re-implementation of DINO based on https://github.com/facebookresearch/dino.

    Args:
        teacher (nn.Module): teacher network instance (default: ViT-Base)
        student (nn.Module): student network instance (default: ViT-Base)
        out_dim (int): output dimension (default: 8192)
        global_crops_number (int): number of global crops (default: 2)
        local_crops_number (int): number of local crops (default: 10)
        student_temperature (float): temperature of student network on cls token (default: 0.1)
        cls_temperature (float): temperature of teacher network on cls token (default: 0.07)
        center_momentum (float): momentum for center vector (default: 0.9)
        teacher_momentum (float): initial momentum for updating teacher (default: 0.996)
        norm_last_layer (bool): whether to normalize the last layer output. (default: False)
        norm_in_head (str): which type of norm layer used in head. If set as None, no norm layer. (default: None)
        act_in_head (type): activation layer in head. (default: gelu) 
    """
    def __init__(self, teacher, student, out_dim=8192, 
                 global_crops_number=2, local_crops_number=10, 
                 student_temperature=0.1, cls_temperature=0.07,
                 center_momentum=0.9, teacher_momentum=0.996, 
                 norm_last_layer=False, norm_in_head=None, act_in_head="gelu",
                 **kwargs):
        super().__init__()
        ## init teacher and student
        self.teacher = teacher
        self.student = student
        
        self.backbone = self.teacher

        ## ensure that the teacher and the student share the same weight
        msg = self.teacher.load_state_dict(self.student.state_dict(), strict=False)
        print("data copying from student to teacher with msg: {}".format(msg))
        for param_teacher in self.teacher.parameters():
            param_teacher.requires_grad = False

        embed_dim = self.student.embed_dim

        self.student = layers.MultiCropWrapper(student, layers.DINOHead(
                embed_dim, out_dim,
                norm=norm_in_head, act=act_in_head,
                norm_last_layer=norm_last_layer,
            ))
        self.teacher = layers.MultiCropWrapper(
            teacher, layers.DINOHead(
                embed_dim, out_dim, 
                norm=norm_in_head, act=act_in_head
            ),
        )

        # create hooks for the embeddings before and after projection
        self.entangled_dim = self.student.head.mlp.weight.shape[1] if isinstance(self.student.head.mlp, nn.Linear) \
            else self.student.head.mlp[0].weight.shape[1]
        self.inst_dim = self.student.head.mlp.weight.shape[0] if isinstance(self.student.head.mlp, nn.Linear) \
            else self.student.head.mlp[-1].weight.shape[0]
        self._embeddings = {}
        def hook_function(module, input_emb, output_emb):
            if isinstance(input_emb, tuple) and len(input_emb) == 1:
                input_emb = input_emb[0]
            if isinstance(output_emb, tuple) and len(output_emb) == 1:
                output_emb = output_emb[0]
            self._embeddings["entangled_cls_emb"] = input_emb
            self._embeddings["inst_cls_emb"] = output_emb
        self.student.head.mlp.register_forward_hook(hook_function)

        self.student_temperature = student_temperature
        self.cls_temperature = cls_temperature

        self.center_momentum = center_momentum
        self.teacher_momentum = teacher_momentum

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.ncrops = global_crops_number + local_crops_number
        self.register_buffer("center", torch.zeros(1, out_dim)) 

    def get_loss(self, student_cls, teacher_cls, student_local_cls, 
        cls_temperature):      
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        student_cls = student_cls / self.student_temperature
        student_cls_c = student_cls.chunk(self.ncrops)
        

        teacher_cls_c = F.softmax((teacher_cls - self.center) / cls_temperature, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.global_crops_number)

        total_loss, n_loss_terms = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    continue
                else:
                    loss = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
            
        total_loss = total_loss / n_loss_terms 
        total_loss = dict(cls_loss=total_loss, loss=total_loss)
        self.update_center(teacher_cls)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls):
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

    @torch.no_grad()
    def momentum_update_teacher(self, momentum=None):
        m = momentum if momentum is not None else self.teacher_momentum
        student = self.student.module if isinstance(self.student, DistributedDataParallel) else self.student
        teacher = self.teacher.module if isinstance(self.teacher, DistributedDataParallel) else self.teacher

        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
        for param_q, param_k in zip(params_q, params_k):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def forward_feature(self, images):
        feature = self.teacher.backbone(images)
        return self.teacher.head.mlp(feature)
        
    def forward(self, imgs, masks=None, global_crops_number=None, 
                    cls_temperature=None, return_hooks=False,
                    teacher_momentum=None):
        """
        Given a list of images and corresponding masks, compute cross-view 
        cluster assignment prediction loss.

        Args:
            imgs (list of torch.Tensor [N,C,H,W]): image tensors. 
            masks (list of torch.Tensor [N,C,H,W]): mask tensors.
            global_crops_number (int, optional): number of global crops. Use 
                self.global_crops_number if set as None. (default: None)
            cls_temperature (float, optional) temperature on teacher cls token output. 
                Use self.cls_temperature if set as None. (default: None)
            return_hooks (bool): whether return the hidden output by hook function. (default: False)
            teacher_momentum (float, optional): momentum for teacher network update.
                Use self.teacher_momentum if set as None. (default: None)

        Returns:
            output_dict: dict with the following fields:
                ``loss`` (torch.Tensor [1,]): loss for backward.
                ``probs_teacher`` (list of torch.Tensor [N, C]): teacher output indicating the 
                    "probability" of cluster assignment.
                ``probs_student`` (list of torch.Tensor [N, C]): student output indicating the 
                    "probability" of cluster assignment.
                ``embeddings``: dict used for further loss computation in hirl.
        """
        ## momentumly update teacher
        self.momentum_update_teacher(teacher_momentum)
        global_crops_number = global_crops_number if global_crops_number is not None else self.global_crops_number
        teacher_output = self.teacher(imgs[:global_crops_number])
        student_output = self.student(imgs[:global_crops_number])

        ## only 1 global crop
        embedding_dict = defaultdict(list)
        batch_size = imgs[0].shape[0]
        for k,v in self._embeddings.items():
            embedding_dict[k].append(v[:batch_size])

        if len(imgs) > global_crops_number:
            student_local_cls = self.student(imgs[global_crops_number:], mask=None)
            for k,v in self._embeddings.items():
                embedding_dict[k].extend(list(v.chunk(self.local_crops_number)))
        else:
            student_local_cls = None
        
        cls_temperature = cls_temperature if cls_temperature is not None else self.cls_temperature

        output_dict = self.get_loss(student_output, teacher_output, student_local_cls, cls_temperature) 
        output_dict.update({"probs_teacher": teacher_output.chunk(global_crops_number), 
                            "probs_student": student_output.chunk(global_crops_number)})
        if return_hooks:
            output_dict["embeddings"] = embedding_dict
 
        return output_dict


def dino_vit_small(drop_path_rate, **kwargs):
    student = vit_small(patch_size=16, return_all_tokens=False, 
                        drop_path_rate=drop_path_rate)
    teacher = vit_small(patch_size=16, return_all_tokens=False)
    model = DINO(teacher, student, **kwargs)
    return model

def dino_vit_base(drop_path_rate, **kwargs):
    student = vit_base(patch_size=16, return_all_tokens=False, 
                      drop_path_rate=drop_path_rate)
    teacher = vit_base(patch_size=16, return_all_tokens=False)
    model = DINO(teacher, student, **kwargs)
    return model

def dino_vit_large(drop_path_rate, **kwargs):
    student = vit_large(patch_size=16, return_all_tokens=False, 
                        drop_path_rate=drop_path_rate)
    teacher = vit_large(patch_size=16, return_all_tokens=False)
    model = DINO(teacher, student, **kwargs)
    return model

