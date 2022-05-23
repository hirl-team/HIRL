import math
import os

import torch

from hirl.models import simsiam as simsiam_models
from hirl.runners import StandardPretrainRunner
from hirl.utils import dist as dist_utils
from hirl.utils import misc
from hirl.utils.misc import ImageFolderInstance
from hirl.utils.transforms import MultiCropsTransform, SingleCropTransform


class SimSiamRunner(StandardPretrainRunner):
    """
    Runnner supporting SimSiam pre-training.
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
        """
        simsiam convert SyncBatchNorm
        """
        model = simsiam_models.__dict__[args.model]()
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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

        ## simsiam specific trick: fix pred lr
        if args.fix_pred_lr:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = model.parameters()

        optimizer = torch.optim.SGD(optim_params, args.lr, 
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optimizer

    def adjust_learning_rate(self, optimizer, epoch, args):
        """Decay the learning rate based on schedule
            Some of the parameters are with fixed learning rate.
        """
        epoch = math.floor(epoch) # epoch might be float, floor it to int (example: 3.8 => 3)
        lr = args.lr
        # cosine lr schedule
        lr = args.min_lr + 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (args.lr - args.min_lr)

        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = args.lr
            else:
                param_group['lr'] = lr
