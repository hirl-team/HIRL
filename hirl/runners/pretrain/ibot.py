import os

import numpy as np
import torch
import torch.nn as nn

from hirl.models import ibot as models_ibot
from hirl.utils import misc
from hirl.utils.misc import ImageFolderMask
from hirl.utils.transforms import DataAugmentationiBOT

from .dino import DINORunner


class iBOTRunner(DINORunner):
    """
    Runner supporting iBOT pre-training.
    The specific tricks follows DINO.
    """
    def build_dataset(self, args):
        transform = DataAugmentationiBOT(
            args.global_crops_scale,
            args.local_crops_scale,
            args.global_crops_number,
            args.local_crops_number,
        )
        pred_size = args.patch_size
        ## the path should be training data
        dataset = ImageFolderMask(
            os.path.join(args.data_path, 'train'), 
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch)

        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.dataset = dataset
        self.data_loader = data_loader
        print(f"Data loaded: there are {len(dataset)} images.")

    def build_model(self, args):
        model = models_ibot.__dict__[args.model](drop_path_rate=args.drop_path_rate, 
                                                 masked_im_modeling=args.masked_im_modeling, 
                                                 out_dim=args.out_dim, patch_out_dim=args.patch_out_dim, 
                                                 global_crops_number=args.global_crops_number, 
                                                 local_crops_number=args.local_crops_number, 
                                                 student_temperature=args.student_temperature, 
                                                 cls_temperature=args.cls_temperature,
                                                 patch_temperature=args.patch_temperature,
                                                 center_momentum=args.center_momentum,
                                                 center_momentum2=args.center_momentum2,
                                                 teacher_momentum=args.teacher_momentum,
                                                 lambda1=args.lambda1,
                                                 lambda2=args.lambda2,
                                                 norm_last_layer=args.norm_last_layer, 
                                                 norm_in_head=args.norm_in_head,
                                                 act_in_head=args.act_in_head, 
                                                 shared_head=args.shared_head,
                                                 shared_head_teacher=args.shared_head_teacher)
        model = model.cuda()
        if misc.has_batchnorms(model.student):
            model.student = nn.SyncBatchNorm.convert_sync_batchnorm(model.student)
            model.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(model.teacher)
        else:
            # teacher_without_ddp and teacher are the same thing
            pass
        
        self.model = nn.parallel.DistributedDataParallel(model,  device_ids=[args.gpu], find_unused_parameters=True)
        self.model_without_ddp = self.model.module

    def build_optimizer(self, args, model=None):
        """
        further add teacher patch tempearture scheduling 
        based on dino.
        """
        super().build_optimizer(args, model)
        if args.pred_start_epoch == 0:
            self.teacher_patch_temperature_schedule = np.concatenate((
                np.linspace(args.warmup_teacher_patch_temperature,
                            args.patch_temperature, 
                            args.warmup_teacher_temperature_epochs),
                np.ones(args.epochs - args.warmup_teacher_temperature_epochs) * args.patch_temperature
            ))
        else:
            self.teacher_patch_temperature_schedule = np.concatenate((
                np.ones(args.pred_start_epoch) * args.warmup_teacher_patch_temperature,
                np.linspace(args.warmup_teacher_patch_temperature,
                            args.patch_temperature, args.warmup_teacher_temperature_epochs),
                np.ones(args.epochs - args.warmup_teacher_temperature_epochs - args.pred_start_epoch) * args.patch_temperature
        ))
        print(f"Loss, optimizer and schedulers ready.")

    def set_epoch(self, epoch):
        """
        ImageFolderMask set epoch
        """
        super().set_epoch(epoch)
        self.data_loader.dataset.set_epoch(epoch)

    def forward_batch(self, batch, **kwargs):
        output_dict = super().forward_batch(batch, 
                                            masks=[msk.cuda(non_blocking=True) for msk in batch["masks"]],
                                            patch_temperature=self.teacher_patch_temperature_schedule[self.epoch],
                                            **kwargs)
        return output_dict



