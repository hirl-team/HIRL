import math
import os

import torch
import torch.nn.functional as F

from hirl.models import mocov3 as models_mocov3
from hirl.runners import StandardPretrainRunner
from hirl.utils import dist as dist_utils
from hirl.utils import misc
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import DataAugmentationMoCoV3


class MoCoV3Runner(StandardPretrainRunner):
    """
    Runner supporting MoCoV3 pre-training.
    """
    def build_dataset(self, args):
        transform = DataAugmentationMoCoV3(args.global_crops_scale, args.local_crops_scale, args.global_crops_number, 
            args.local_crops_number)
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
        local_views_sum = args.get("local_views_sum", False)
        model = models_mocov3.__dict__[args.model](local_views_sum=local_views_sum)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
        self.model = model
        self.model_without_ddp = model.module

    def build_optimizer(self, args, model=None):
        model = self.model_without_ddp if model is None else model

        eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
        args.lr = args.lr * eff_batch_size / 256
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.optimizer == 'lars':
            optimizer = misc.LARS(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()
        self.optimizer = optimizer

    def adjust_learning_rate(self, optimizer, epoch, args):
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs 
        else:
            lr = args.min_lr + \
                0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))) * (args.lr - args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_moco_momentum(self):
        """Adjust moco momentum based on current epoch"""
        epoch = self.it / self.iter_per_epoch
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.args.epochs)) * (1. - self.args.moco_momentum)
        return m

    def forward_batch(self, batch, **kwargs):
        images = batch["images"]
        if not isinstance(images, list):
            images = [images]
        images = [image.cuda(non_blocking=True) for image in images]
        with torch.cuda.amp.autocast(self.args.use_fp16):
            cosine_momentum = self.args.get("moco_momentum_cosine", True)
            if cosine_momentum:
                m = self.adjust_moco_momentum()
            else:
                m = self.args.moco_momentum
            output_dict = self.model(images, m=m, **kwargs)

        if "targets" in batch:
            image_labels = batch["targets"]
            output_dict["image_labels"] = image_labels
        return output_dict

    def log_metrics(self, metric_logger, output_dict):
        logits, targets = output_dict["logits"], output_dict["targets"]
        local_logits, local_targets = output_dict["local_logits"], output_dict["local_targets"]
        
        for vid, (global_logit, global_target, local_logit, local_target) in enumerate(zip(logits, targets, local_logits, local_targets)):
            loss_global = F.cross_entropy(global_logit, global_target).item()
            acc_global = misc.accuracy(global_logit, global_target)[0].item()
            metric_logger.update(**{f"loss_global_{vid}": loss_global})
            metric_logger.update(**{f"acc_global_{vid}": acc_global})

            if local_logit is not None:
                loss_local = 0.
                for local_id, (local_l, local_t) in enumerate(zip(local_logit, local_target)):
                    loss_local_ = F.cross_entropy(local_l, local_t).item()
                    loss_local += loss_local_
                    metric_logger.update(**{f"loss_local_{local_id}-{vid}": loss_local_})
                    acc_local = misc.accuracy(local_l, local_t)[0].item()
                    metric_logger.update(**{f"acc_local_{local_id}-{vid}": acc_local})
                metric_logger.update(**{f"loss_local_{vid}": loss_local})
