import math
import os

import torch
import torch.nn.functional as F

from hirl.models import beit as models_beit
from hirl.runners import StandardPretrainRunner
from hirl.utils import dist as dist_utils
from hirl.utils import misc
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import DataAugmentationBEiT


class BEiTRunner(StandardPretrainRunner):
    """
    Runner supporting BEiT pre-training.
    """
    def build_dataset(self, args):
        transform = DataAugmentationBEiT(args)
        dataset = ImageFolderInstance(os.path.join(args.data_path, "train"), transform=transform)
        print(dataset)

        num_tasks = dist_utils.get_world_size()
        global_rank = dist_utils.get_rank()
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Training sampler = %s" % str(sampler))

        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        self.dataset = dataset
        self.data_loader = data_loader

    def build_model(self, args):
        model = models_beit.__dict__[args.model]()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        self.model = model
        self.model_without_ddp = model.module

    def build_optimizer(self, args, model=None):
        # Build optimizer
        model = self.model_without_ddp if model is None else model

        skip_list = getattr(model.vit_model, "no_weight_decay", {})
        if len(skip_list) > 0:
            skip_list = {"vit_model." + name for name in skip_list}
        params_groups = misc.get_params_groups(model, skip_list=skip_list)
        if args.optimizer == "adamw":
            optim_args = dict(lr=args.lr, weight_decay=args.weight_decay, eps=1e-8, betas=[0.9, 0.999])
            optimizer = torch.optim.AdamW(params_groups, **optim_args)
        else:
            raise NotImplementedError()

        # Build schedulers for learning rate and weight decay
        lr_schedule = misc.cosine_scheduler(
            args.lr, args.min_lr,
            args.epochs, len(self.data_loader),
            warmup_epochs=args.warmup_epochs
        )
        weight_decay_end = args.get("weight_decay_end", args.weight_decay)
        wd_schedule = misc.cosine_scheduler(
            args.weight_decay, weight_decay_end,
            args.epochs, len(self.data_loader)
        )

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule

    def adjust_learning_rate(self, optimizer, epoch, args):
        iter_per_epoch = args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.data_loader) if iter_per_epoch < 0 else iter_per_epoch

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[int(epoch * iter_per_epoch)]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[int(epoch * iter_per_epoch)]

    def log_metrics(self, metric_logger, output_dict):
        logit, target = output_dict["logit"], output_dict["target"]
        mim_acc = (logit.max(-1)[1] == target).float().mean().item()
        metric_logger.update(mim_acc=mim_acc)
