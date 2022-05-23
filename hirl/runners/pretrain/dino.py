import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from hirl.models import dino as models_dino
from hirl.runners import StandardPretrainRunner
from hirl.utils import misc
from hirl.utils import dist as dist_utils
from hirl.utils.clustering import eval_pred
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import DataAugmentationDINO


class DINORunner(StandardPretrainRunner):
    """
    Runner supporting DINO pre-training.
    some specific tricks in this runner (reference: https://github.com/facebookresearch/dino):
        scheduling:
            - scheduled teacher cls token temperature
            - scheduled teacher momentum
            - scheduled weight decay
        gradient clipping
            freeze last layers at the first epoch.
    """
    def __init__(self, args, resume=True):
        super().__init__(args, resume)
        self.buffer_dict = defaultdict(list)

    def build_dataset(self, args):
        transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.global_crops_number,
            args.local_crops_number,
        )
        dataset = ImageFolderInstance(os.path.join(args.data_path, "train"), 
                                        transform=transform)

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
        # ============ building student and teacher networks ... ============
        model = models_dino.__dict__[args.model](drop_path_rate=args.drop_path_rate, 
                                                 out_dim=args.out_dim,
                                                 global_crops_number=args.global_crops_number, 
                                                 local_crops_number=args.local_crops_number, 
                                                 student_temperature=args.student_temperature, 
                                                 cls_temperature=args.cls_temperature,
                                                 center_momentum=args.center_momentum,
                                                 teacher_momentum=args.teacher_momentum,
                                                 norm_last_layer=args.norm_last_layer, 
                                                 norm_in_head=args.norm_in_head,
                                                 act_in_head=args.act_in_head)
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
        model = self.model_without_ddp if model is None else model

        eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
        
        args.lr = args.lr * eff_batch_size / 256

        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)
        params_groups = misc.get_params_groups(model)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups, lr=args.lr)  # to use with ViTs
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=args.lr, momentum=0.9)  # lr is set by scheduler
        elif args.optimizer == "lars":
            optimizer = misc.LARS(params_groups)  # to use with convnet and large batches
        else:
            raise NotImplementedError()

        # ============ init schedulers ... ============
        lr_schedule = misc.cosine_scheduler(
            args.lr, args.min_lr,
            args.epochs, len(self.data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule = misc.cosine_scheduler(
            args.weight_decay, args.weight_decay_end,
            args.epochs, len(self.data_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = misc.cosine_scheduler(args.teacher_momentum, 1,
                                                args.epochs, len(self.data_loader))

        ## teacher cls temperature follows linear warmup schedule
        ## specifically, 0.04 => 0.07 in 30 epochs.
        self.teacher_cls_temperature_schedule = np.concatenate((
            np.linspace(args.warmup_teacher_cls_temperature,
                        args.cls_temperature, 
                        args.warmup_teacher_temperature_epochs),
            np.ones(args.epochs - args.warmup_teacher_temperature_epochs) * args.cls_temperature
        ))

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        print(f"Loss, optimizer and schedulers ready.")

    def forward_batch(self, batch, **kwargs):
        output_dict = super().forward_batch(batch, 
        cls_temperature=self.teacher_cls_temperature_schedule[self.epoch],
        teacher_momentum=self.momentum_schedule[self.it], **kwargs)
        return output_dict

    def log_metrics(self, metric_logger, output_dict):
        # log statistics (this part could be utilized to monitor the convergence of training.)
        probs1 = output_dict.pop("probs_teacher") # teacher cls token [N, D]
        probs2 = output_dict.pop("probs_student") # student cls token [N, D]
        labels = output_dict.pop("image_labels") # image labels [N, ]
        pred1 = misc.concat_all_gather(probs1[0].max(dim=1)[1]) # teacher cls token hard prediction [N, ]
        pred2 = misc.concat_all_gather(probs2[1].max(dim=1)[1]) # student cls token hard prediction [N, ]
        acc = (pred1 == pred2).sum() / pred1.size(0)

        self.buffer_dict["pred_labels"].append(pred1)
        self.buffer_dict["real_labels"].append(misc.concat_all_gather(labels.to(pred1.device)))

        # logging
        torch.cuda.synchronize()

        metric_logger.update(wd=self.optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    def step_optimizer(self, loss, it, accum_iter, clip_grad):
        """
        diff from standard: fix the last layer before freeze_last_layer
        """
        fixed_parameters = None
        ## fix last layer gradient (ibot trick)
        if self.epoch >= self.args.freeze_last_layer:
            pass
        else:
            fixed_parameters = list()
            for n, p in self.model.named_parameters():
                if "last_layer" in n:
                    fixed_parameters.append(p)

        loss = loss / accum_iter
        self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                    update_grad=(it + 1) % accum_iter == 0, clip_grad=clip_grad, 
                    fixed_parameters=fixed_parameters)
        if (it + 1) % accum_iter == 0:
            self.optimizer.zero_grad()

    def eval_after_epoch(self):
        pred_labels = self.buffer_dict["pred_labels"]
        real_labels = self.buffer_dict["real_labels"]

        pred_labels = torch.cat(pred_labels).cpu().detach().numpy() # [N, ] CLS token predicted index 
        real_labels = torch.cat(real_labels).cpu().detach().numpy() # [N, ] real imagenet label
        nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)

        print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))

        self.buffer_dict["pred_labels"] = list()
        self.buffer_dict["real_labels"] = list()
        return {"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc}

    def adjust_learning_rate(self, optimizer, epoch, args):
        iter_per_epoch = args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.data_loader) if iter_per_epoch < 0 else iter_per_epoch

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[int(epoch * iter_per_epoch)]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[int(epoch * iter_per_epoch)]
