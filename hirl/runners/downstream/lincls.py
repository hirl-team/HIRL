import math
import sys

import torch
import torch.nn as nn
from hirl.backbones import vision_transformer as vt
from hirl.runners import StandardDownstreamRunner
from hirl.utils import misc
from torchvision import transforms
from torchvision.models import resnet


class LinclsRunner(StandardDownstreamRunner):
    """
    ImageNet linear classification runner.
    """
    filtered_keys = []

    def build_transform(self, is_train, args):
        if is_train:
            return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        else:
            return transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def build_model(self, args):
        arch_kwargs = args.get("arch_kwargs", dict())
        if args.arch in vt.__dict__.keys():
            model = vt.__dict__[args.arch](
                patch_size=args.patch_size,
                use_mean_pooling=args.use_mean_pooling,
                drop_path_rate=args.drop_path_rate,
                num_classes=args.num_classes,
                use_head=True, **arch_kwargs
            )
            ## special head dimension for n_last_blocks
            if args.get("n_last_blocks", 0):
                embed_dim = model.embed_dim * (args.n_last_blocks + int(args.use_mean_pooling))
                del model.head
                model.head = nn.Linear(embed_dim, args.num_classes)
        elif args.arch in resnet.__dict__.keys():
            model = resnet.__dict__[args.arch](num_classes=args.num_classes)
        else:
            raise NotImplementedError

        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                find_unused_parameters=True)
        model_without_ddp = model.module

        self.model = model
        self.model_without_ddp = model_without_ddp

        ## lincls specific: fix trunk
        self.fix_trunk(args)
        self.build_criterion(args)

    def fix_trunk(self, args):
        for name, param in self.model_without_ddp.named_parameters():
            param.requires_grad = False

        if "resnet" in args.arch: #resnet 
            for param in self.model_without_ddp.fc.parameters():
                param.requires_grad = True
            ## init fc layer (used in mocov2)
            self.model_without_ddp.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.model_without_ddp.fc.bias.data.zero_()
        elif "vit" in args.arch:
            for param in self.model_without_ddp.head.parameters():
                param.requires_grad = True
            self.model_without_ddp.head.weight.data.normal_(mean=0.0, std=0.01)
            self.model_without_ddp.head.bias.data.zero_()

    def build_criterion(self, args):
        criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion
    
    def train_one_epoch(self, epoch):
        self.model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
        log_interval = self.args.get("log_interval", 10)
        iter_per_epoch = self.args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.loader_train) if iter_per_epoch < 0 else iter_per_epoch
    
        accum_iter = self.args.get("accum_iter", 1)
        clip_grad = self.args.get("clip_grad", None)

        self.optimizer.zero_grad()

        for it, (samples, targets) in enumerate(metric_logger.log_every(self.loader_train, log_interval, header)):
            if it >= iter_per_epoch:
                break

            it = len(self.loader_train) * epoch + it  # global training iteration
            # we use a per iteration (instead of per epoch) lr scheduler
            if it % accum_iter == 0: # TODO: merge the code style
                self.adjust_learning_rate(self.optimizer, it / len(self.loader_train), self.args)

            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(self.args.use_fp16):
                if "vit" in self.args.arch and self.args.n_last_blocks:
                    intermediate_out = self.model_without_ddp.get_intermediate_layers(samples, self.args.n_last_blocks)
                    outputs = torch.cat([x[:, 0] for x in intermediate_out], dim=-1)
                    if self.args.use_mean_pooling:
                        outputs = torch.cat((outputs.unsqueeze(-1), torch.mean(intermediate_out[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        outputs = outputs.reshape(outputs.shape[0], -1)
                    outputs = self.model_without_ddp.head(outputs)
                else: # normal inference
                    outputs = self.model(samples)

                loss = self.criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / accum_iter
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                        update_grad=(it + 1) % accum_iter == 0, clip_grad=clip_grad)
            if (it + 1) % accum_iter == 0:
                self.optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def adjust_learning_rate(self, optimizer, epoch, args):
        if args.cos:
            if epoch < args.warmup_epochs:
                lr = args.lr * epoch / args.warmup_epochs 
            else:
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            lr = args.lr
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.

        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
