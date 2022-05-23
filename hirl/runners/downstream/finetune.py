import math
import sys

import PIL
import torch
from hirl.runners import StandardDownstreamRunner
from hirl.utils import misc
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import transforms


class FinetuneRunner(StandardDownstreamRunner):
    """
    ImageNet fine-tuning runner.
    """
    filtered_keys = []

    def build_transform(self, is_train, args):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # train transform
        if is_train:
            use_timm_aug = args.get("timm_aug", True)
            if use_timm_aug:
                # this should always dispatch to transforms_imagenet_train
                transform = create_transform(
                    input_size=args.input_size,
                    is_training=True,
                    color_jitter=args.color_jitter,
                    auto_augment=args.auto_aug_policy,
                    interpolation='bicubic', 
                    re_prob=args.reprob,
                    re_mode=args.remode,
                    re_count=args.recount,
                    mean=mean,
                    std=std,
                )
            else:
                ## for mocov2, simsiam and swav
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
                transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,])
            return transform
        else:
            # eval transform
            t = []
            if args.input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(args.input_size / crop_pct)
            t.append(
                transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            return transforms.Compose(t)

    def build_criterion(self, args):
        """
        with mixup enabled.
        """
        ## build criterion
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))
        print("mixup function: {}".format(mixup_fn))
        self.criterion = criterion
        self.mixup_fn = mixup_fn

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
            if it % accum_iter == 0:
                self.adjust_learning_rate(self.optimizer, it / len(self.loader_train), self.args)

            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)

            with torch.cuda.amp.autocast(self.args.use_fp16):
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
        """Decay the learning rate with half-cycle cosine after warmup (from MAE)
        """
        use_cos = args.get("cos", True)
        if use_cos:
            if epoch < args.warmup_epochs:
                start_lr = args.get("start_lr", None)
                if start_lr:
                    lr = start_lr + (args.lr - start_lr) * epoch / args.warmup_epochs
                else:
                    lr = args.lr * epoch / args.warmup_epochs 
            else:
                lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.get("schedule", None):
            lr = args.lr
            for milestone in args.schedule:
                lr *= args.gamma if epoch >= milestone else 1.
        else:
            raise NotImplementedError
        # for vit layer decay
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
