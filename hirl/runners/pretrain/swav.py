import math
import os

import torch
import torch.nn as nn
from apex.parallel.LARC import LARC

from hirl.models import swav as models_swav
from hirl.runners import StandardPretrainRunner
from hirl.utils import dist as dist_utils
from hirl.utils import misc
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import MultiCropsTransform, SingleCropTransform


class SwAVRunner(StandardPretrainRunner):
    """
    Runner supporting SwAV pre-training.
    """
    def build_dataset(self, args):
        color_scale = args.get("color_scale", 1.0)
        color_first = args.get("color_first", False)        
        transform_train = MultiCropsTransform(args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, 
                          color_scale=color_scale, color_first=color_first) \
            if args.multi_crop else SingleCropTransform(args.size_crops[0], args.min_scale_crops[0], args.max_scale_crops[0], 
                          color_scale=color_scale, color_first=color_first)
        dataset = ImageFolderInstance(os.path.join(args.data_path, 'train'), transform=transform_train)
        print("transformation:", transform_train)
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
        model = models_swav.__dict__[args.model](batch_size=args.batch_size, queue_length=args.queue_length)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
        if args.optimizer=="lars":
            print("Using LARS optimizer")
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        print("optimizer: ", optimizer)
        self.optimizer = optimizer

    def adjust_learning_rate(self, optimizer, epoch, args):
        """
        Iteration-wise cosine lr scheduling with warmup.
        """
        warmup_epoch = args.get("warmup_epochs", 0.)
        start_lr = args.get("start_lr", 0.)
        lr = args.lr
        if epoch < warmup_epoch:
            lr = start_lr + (args.lr - start_lr) * epoch / warmup_epoch
        else:
            lr = args.min_lr + \
                0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoch) / (args.epochs - warmup_epoch))) * (args.lr - args.min_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # freeze prototypes at the start of training
        iter_per_epoch = args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.data_loader) if iter_per_epoch < 0 else iter_per_epoch
        freeze_proto_iter = args.get("freeze_proto_iter", 0.) / iter_per_epoch
        if epoch < freeze_proto_iter:
            for name, p in self.model_without_ddp.named_parameters():
                if "prototypes" in name:
                    p.requires_grad = False
                    print(f"current epoch is {epoch} | before {freeze_proto_iter}, layer {name} no grad.")
        else:
            for name, p in self.model_without_ddp.named_parameters():
                p.requires_grad = True

    def log_metrics(self, metric_logger, output_dict):
        src_logit = output_dict["source_logits"]
        tgt_logit = output_dict["target_logits"]
        acc = (src_logit.argmax(dim=-1) == tgt_logit.argmax(dim=-1)).float().mean().item()
        metric_logger.update(acc=acc)
