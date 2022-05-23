import math
import os

import torch
import torch.nn.functional as F

from hirl.models import moco as models_moco
from hirl.runners import StandardPretrainRunner
from hirl.utils import dist as dist_utils
from hirl.utils import misc
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import MultiCropsTransform, SingleCropTransform


class MoCoRunner(StandardPretrainRunner):
    """
    Runner supporting MoCoV2 pre-training.
    """
    def build_dataset(self, args):
        transform_train = MultiCropsTransform(args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops) \
            if args.multi_crop else SingleCropTransform(args.size_crops[0], args.min_scale_crops[0], args.max_scale_crops[0])
        dataset = ImageFolderInstance(os.path.join(args.data_path, 'train'), transform=transform_train)
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
        model = models_moco.__dict__[args.model]()
        model = model.cuda()
        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        self.model = model
        self.model_without_ddp = model_without_ddp
    
    def build_optimizer(self, args, model=None):
        model = self.model_without_ddp if model is None else model

        eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
        args.lr = args.lr * eff_batch_size / 256
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optimizer

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule"""
        epoch = math.floor(epoch) # epoch might be float, floor it to int (example: 3.8 => 3)
        lr = args.lr
        # cosine lr schedule
        lr = args.min_lr + 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (args.lr - args.min_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def log_metrics(self, metric_logger, output_dict):
        logits, targets = output_dict["logits"], output_dict["targets"]
        local_logits, local_targets = output_dict["local_logits"], output_dict["local_targets"]

        loss_global = F.cross_entropy(logits, targets).item()
        metric_logger.update(loss_global=loss_global)
        acc_global = misc.accuracy(logits, targets)[0].item()
        metric_logger.update(acc_global=acc_global)

        if local_logits is not None:
            loss_local = 0.
            for vid, (local_logit, local_target) in enumerate(zip(local_logits, local_targets)):
                loss_local_ = F.cross_entropy(local_logit, local_target).item()
                loss_local += loss_local_
                metric_logger.update(**{f"loss_local{vid}": loss_local_})
                acc_local = misc.accuracy(local_logit, local_target)[0].item()
                metric_logger.update(**{f"acc_local{vid}": acc_local})
            metric_logger.update(loss_local=loss_local)
