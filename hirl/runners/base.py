import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import hirl.models as pretrain_models
import hirl.utils.dist as dist_utils
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as torchvision_dataset
from addict import Dict as adict
from apex.parallel.LARC import LARC
from hirl import utils
from hirl.backbones import vision_transformer as vt
from hirl.backbones.vision_transformer import VisionTransformer, trunc_normal_
from hirl.utils import misc
from hirl.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.utils import accuracy
from torchvision import datasets
from torchvision.models import resnet


class BaseRunner(object):
    def __init__(self, args: adict, resume=True) -> None:
        super().__init__()
        self.args = args
        dist_utils.init_distributed_mode(self.args)
        self.local_rank = args.local_rank
        self.rank = dist_utils.get_rank()
        misc.fix_random_seeds(self.args.seed)
        cudnn.benchmark = True
        
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(args.to_dict().items())))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.custom_initialize()

        self.build_dataset(self.args)
        self.build_model(self.args)
        self.build_optimizer(self.args)
        self.build_scaler(self.args)
        if resume:
            self.resume(self.args)

    def custom_initialize(self):
        """
        we use this function to perform auto-resume from save dir.
        """

        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.autoresume and os.path.exists(os.path.join(self.args.output_dir, "latest.pth")):
            self.args.resume = os.path.join(self.args.output_dir, "latest.pth")

    def custom_end(self):
        pass

    def build_dataset(self, args):
        raise NotImplementedError

    def build_model(self, args):
        raise NotImplementedError

    def build_optimizer(self, args):
        raise NotImplementedError

    def build_scaler(self, args):
        raise NotImplementedError

    def resume(self, args):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def save_model(self, save_dict, epoch):
        torch.save(save_dict, os.path.join(self.args.output_dir, 'latest.pth'))
        if  (epoch % self.args.save_interval == 0) and epoch:
            torch.save(save_dict, os.path.join(self.args.output_dir, f'checkpoint_{epoch:04}.pth'))

    def restart_from_checkpoint(self, ckp_path, run_variables=None, **kwargs):
        """
        Re-start from checkpoint; 
        load the key into the value of kwargs
        kwargs in the format: 
            { key (str): value(nn.Module)}
        """
        if not os.path.isfile(ckp_path):
            return
        print("Found checkpoint at {}".format(ckp_path))

        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location="cpu")

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                    except ValueError:
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
            else:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

        # re load variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]
    
    def log_metrics(self, metric_logger, output_dict):
        pass

class StandardPretrainRunner(BaseRunner):
    """
    A standard pretrianing runner with amp loss scaler supported and some basic running functions.
    """
    def to_ddp(self, model, args):
        """
        some runners may have specific to_ddp operation. 
        For example, iBOT and DINO...
        """
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        self.model_without_ddp = model_without_ddp
        return model

    def build_dataset(self, args):
        transform_class = args.dataset.transform.pop("name")
        transform_args = args.dataset.pop("transform")
        train_transform = utils.transforms.__dict__[transform_class](**transform_args)

        dataset_class = args.dataset.pop("name")
        ## first find it in our utils, then torchvision
        if dataset_class in utils.loader.__dict__:
            dataset =  utils.loader.__dict__[dataset_class](transform=train_transform, **args.dataset)
        elif dataset_class in torchvision_dataset.__dict__:
            dataset = torchvision_dataset.__dict__[dataset_class](transform=train_transform, **args.dataset)
        else:
            raise NotImplementedError("dataset class {} not found.".format(dataset_class))

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
        a general model building function
        """
        model_class = args.model.pop("name")
        model = pretrain_models.__dict__[model_class](**args.model)
        model = self.to_ddp(model, args)
        self.model = model

    def build_optimizer(self, args):
        """
        a general and standard optimizer building function
        """
        eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
        
        args.lr = args.lr * eff_batch_size / 256

        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        optimizer_class = args.optimizer.name
        optimizer_params = args.optimizer.params

        param_groups = self.model.parameters()
        optimizer_class = args.optimizer.pop("name")
        optimizer = getattr(torch.optim, optimizer_class)(param_groups, **optimizer_params)

        self.optimizer = optimizer

    def build_scaler(self, args):
        """
        support for auto mix precision (amp)
        """
        use_fp16 = args.get("use_fp16", False)
        args.use_fp16 = use_fp16
        if args.use_fp16:
            print("auto mix precision enabled!")
        loss_scaler = NativeScaler(args.use_fp16)
        self.loss_scaler = loss_scaler

    def resume(self, args):
        to_restore = {"epoch": 0}
        if isinstance(args.resume, str) and os.path.exists(args.resume):
            
            self.restart_from_checkpoint(args.resume,
                run_variables=to_restore,
                model=self.model, 
                optimizer=self.optimizer,
                loss_scaler=self.loss_scaler
            )
        start_epoch = to_restore["epoch"]
        self.start_epoch = start_epoch

    def set_epoch(self, epoch):
        """
        set epochs before training. By default, set the 
        distributed sampler epoch.
        """
        if hasattr(self.model, "module"): ## ddp model
            if hasattr(self.model.module, "set_epoch"):
                self.model.module.set_epoch(epoch)
        else:
            if hasattr(self.model, "set_epoch"): # no ddp model
                self.model.set_epoch(epoch)
        self.data_loader.sampler.set_epoch(epoch)
        self.epoch = epoch

    def set_iter(self, it):
        self.it = it

    def set_iter_per_epoch(self, iter_per_epoch):
        self.iter_per_epoch = iter_per_epoch

    @property
    def custom_save_dict(self):
        return {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "loss_scaler": self.loss_scaler.state_dict(),
        }

    def run(self):
        start_time = time.time()
        print(f"Start training for {self.args.epochs} epochs")
        self.model.train()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch)
            # save model
            save_dict = {
                'epoch': epoch + 1,
                'args': self.args.to_dict(),
            }
            save_dict.update(self.custom_save_dict)

            self.save_model(save_dict, epoch)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            if dist_utils.is_main_process():
                with (Path(self.args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def adjust_learning_rate(self, optimizer, epoch, args):
        raise NotImplementedError
    
    def forward_batch(self, batch, **kwargs):
        """
        a standard batch forwarding function. This 
        could be used for CNN-based pre-training (moco, swav, simsiam)
        """

        images = batch["images"]
        if not isinstance(images, list):
            images = [images]
        images = [image.cuda(non_blocking=True) for image in images]
        with torch.cuda.amp.autocast(self.args.use_fp16):
            output_dict = self.model(images, **kwargs)

        if "targets" in batch:
            image_labels = batch["targets"]
            output_dict["image_labels"] = image_labels
        return output_dict

    def eval_after_epoch(self):
        return dict()

    def step_optimizer(self, loss, it, accum_iter, clip_grad):
        """
        update model parameter according to iteration

        Args:
            loss: Tensor
            it (int): current iteration (total)
            accum_iter (int): how many iterations to accumulate the gradient
            clip_grad: (float / None): if has value, clip gradients.
        """
        loss = loss / accum_iter
        self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                    update_grad=(it + 1) % accum_iter == 0, clip_grad=clip_grad, )
        if (it + 1) % accum_iter == 0:
            self.optimizer.zero_grad()

    def train_one_epoch(self, epoch):
        """
        A standard train_one_epoch function including:
        - clip grad
        - gradient accumulation
        - iter-wise learning rate adjust
        - metric logging
        """
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
        log_interval = self.args.get("log_interval", 10)
        iter_per_epoch = self.args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.data_loader) if iter_per_epoch < 0 else iter_per_epoch
        self.set_iter_per_epoch(iter_per_epoch)
        accum_iter = self.args.get("accum_iter", 1)
        clip_grad = self.args.get("clip_grad", None)

        self.model.train()
        for it, (batch) in enumerate(metric_logger.log_every(self.data_loader, log_interval, header)):
            if it >= iter_per_epoch:
                break
            it = iter_per_epoch * epoch + it  # global training iteration
            self.set_iter(it)
            self.adjust_learning_rate(self.optimizer, it / iter_per_epoch, self.args)
            output_dict = self.forward_batch(batch)

            loss = output_dict["loss"]
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                sys.exit(1)

            self.step_optimizer(loss, it, accum_iter, clip_grad)
    
            torch.cuda.synchronize()
            # record training status
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            
            ## specific logging
            self.log_metrics(metric_logger, output_dict)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        ## maybe eval after epoch
        eval_results = self.eval_after_epoch()
        return_dict.update(eval_results)
        return return_dict

class StandardDownstreamRunner(BaseRunner):
    """
    a standard downstream runner base class for:
    - linear evaluation
    - finetune
    - semi-supervised learning
    """
    filtered_keys = []

    def custom_end(self):
        """
        save metric file.
        """
        if self.local_rank == 0:
            local_metric_file = os.path.join(self.args.output_dir, "metric.json")
            metric_dict = {"acc": self.metric}
            json_string = json.dumps(metric_dict)
            with open(local_metric_file, "w") as f:
                f.write(json_string)

    def build_transform(self, is_train, args):
        raise NotImplementedError

    def build_train_dataset(self, args, transform):
        return datasets.ImageFolder(os.path.join(args.data_path, "train"), 
                                        transform=transform)
    
    def build_val_dataset(self, args, transform):
            return datasets.ImageFolder(os.path.join(args.data_path, "val"), 
                                                    transform=transform)
        
    def build_dataset(self, args):
        """
        downstream dataset has its training and valid dataset. 
        """
        transform_train = self.build_transform(True, args)
        transform_val = self.build_transform(False, args)

        dataset_train = self.build_train_dataset(args, transform_train)
        dataset_val = self.build_val_dataset(args, transform_val)
        
        sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)

        if args.dist_eval:
            if len(dataset_val) % dist_utils.get_world_size() != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        loader_train = torch.utils.data.DataLoader(
                        dataset_train, sampler=sampler_train,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True, drop_last=True,
                    )
        loader_val = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=int(args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=True, drop_last=False
                )
        
        self.dataset_train = dataset_train
        self.dataset_val  = dataset_val
        self.loader_train = loader_train
        self.loader_val = loader_val

    def build_model(self, args):
        """
        specific tricks:
        - use_syncbn: set True for swav/hirl/mocov2...semisup evaluation.
        """
        if args.arch in vt.__dict__.keys():
            model = vt.__dict__[args.arch](
                patch_size=args.patch_size,
                use_mean_pooling=args.use_mean_pooling,
                drop_path_rate=args.drop_path_rate,
                num_classes=args.num_classes,
                use_head=True
            )
        elif args.arch in resnet.__dict__.keys():
            model = resnet.__dict__[args.arch](num_classes=args.num_classes)
        else:
            raise NotImplementedError
        model = model.cuda()

        use_syncbn = args.get("use_syncbn", False)
        if use_syncbn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        self.model = model
        self.model_without_ddp = model_without_ddp

        self.build_criterion(args)

    def build_criterion(self, args):
        """
        build criterion is called after build model. 
        For a standard downstream runner, just use nn.CrossEntropy

        """
        criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion

    def build_optimizer(self, args):
        """
        a standard optimizer building function that support 
        every optimizer. Some specific tricks supported:
        - layer decay (MAE, iBOT finetune): support applying 
            layer-wise learning rate decay on the optimizer
        - no weight decay on bias
        - different learning rate from trunk and head
        """
        ## batch size linear scaling rule
        eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()
        args.lr = args.lr * eff_batch_size / 256

        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        optimizer_class = args.optimizer.name
        optimizer_params = args.optimizer.params
        
        layer_decay = args.optimizer.layer_decay

        if layer_decay:
            print("enable layer-wise learning rate decay with a factor of {}".format(layer_decay))
            no_weight_decay_list = getattr(self.model_without_ddp, "no_weight_decay", [])
            param_groups = misc.param_groups_lrd(self.model_without_ddp, 
            args.optimizer.params.weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=layer_decay
            )
        elif args.optimizer.no_weight_decay_on_bias:
            print("no weight decay on bias enabled!")
            param_groups = misc.get_params_groups(self.model_without_ddp)
        elif args.lr_last_layer: # note: this  lr is not scaled according to batch size.
            ## if pecified different lr, apply it to head/fc
            trunk_params, head_params = list(), list()
            for name, param in self.model_without_ddp.named_parameters():
                if "fc" in name or "head" in name:
                    head_params.append(param)
                else:
                    trunk_params.append(param)
            param_groups = [{"params": trunk_params}, {"params": head_params, "lr": args.lr_last_layer}]

        else:
            param_groups = self.model_without_ddp.parameters()


        optimizer = getattr(torch.optim, optimizer_class)(param_groups, lr=args.lr, 
                            **optimizer_params)

        if args.optimizer.use_lars:
            print("use lars optimizer")
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        print("optimizer: {}".format(optimizer))
        self.optimizer = optimizer

    def build_scaler(self, args):
        """
        support for auto mix precision (amp)
        """
        use_fp16 = args.get("use_fp16", False)
        args.use_fp16 = use_fp16
        if args.use_fp16:
            print("auto mix precision enabled!")
        loss_scaler = NativeScaler(args.use_fp16)
        self.loss_scaler = loss_scaler

    def load_pretrain(self, args):
        """
        load pretrain from a checkpoint.
        """
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        model_prefix = args.get("model_prefix", "model")
        checkpoint_model = checkpoint[model_prefix]
        ## automatically remove ddp prefix
        if all([k.startswith("module.") for k in checkpoint_model.keys()]):
            print("remove ddp prefix from model.")
            checkpoint_model = {k.replace("module.", ""):v for k,v in checkpoint_model.items()}
        
        print("backbone prefix: {}".format(args.backbone_prefix))
        if args.backbone_prefix:
            checkpoint_model = {k[len(args.backbone_prefix)+1:]:v for k,v in checkpoint_model.items() if k.startswith(args.backbone_prefix)}
        
        state_dict = self.model_without_ddp.state_dict()
        ## remove head / fc
        removed_keys = list()
        for key in checkpoint_model.keys():
            if key not in state_dict or key in self.filtered_keys or checkpoint_model[key].shape != state_dict[key].shape:
                removed_keys.append(key)

        print("removed keys in pretrained model: {}".format(removed_keys))
        for key in removed_keys:
            checkpoint_model.pop(key)

        if isinstance(self.model_without_ddp, VisionTransformer):
            misc.interpolate_pos_embed(self.model_without_ddp, checkpoint_model)

        msg = self.model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print("loading message: {}".format(msg))

        for param1, param2 in zip(self.model.parameters(), self.model_without_ddp.parameters()):
            if not torch.allclose(param1, param2):
                print("unmatch between model and model_without_ddp detected!!!")

        if isinstance(self.model_without_ddp, VisionTransformer):
            if args.trunc_normal:
                print("manually initialize the fc layer")
                trunc_normal_(self.model_without_ddp.head.weight, std=2e-5)

    def resume(self, args):
        """
        For a downstream /finetune runner. The resume function 
        should also act as a pretrain loading function.
        """
        ## load pretrain
        if os.path.exists(args.pretrained):
            self.load_pretrain(args)

        to_restore = {"epoch": 0}
        if isinstance(args.resume, str) and os.path.exists(args.resume):
            self.restart_from_checkpoint(args.resume,
                run_variables=to_restore,
                model=self.model, 
                optimizer=self.optimizer,
                loss_scaler=self.loss_scaler
            )
        start_epoch = to_restore["epoch"]
        self.start_epoch = start_epoch

    def set_epoch(self, epoch):
        """
        set epochs before training. By default, set the 
        distributed sampler epoch.
        """
        self.loader_train.sampler.set_epoch(epoch)

    @property
    def custom_save_dict(self):
        return {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "loss_scaler": self.loss_scaler.state_dict(),
        }

    def run(self):
        """
        A standard running process for imagenet downstream tasks.
        """
        start_time = time.time()
        max_accuracy = 0.0
        print(f"Start training for {self.args.epochs} epochs")
        self.model.train()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate()
            # save model
            save_dict = {
                'epoch': epoch + 1,
                'args': self.args.to_dict(),
            }
            save_dict.update(self.custom_save_dict)

            self.save_model(save_dict, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch}
            
            if dist_utils.is_main_process():
                with (Path(self.args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            ## best logging
            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
            
            print(f'Max accuracy: {max_accuracy:.2f}%')
                
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.metric = max_accuracy
        self.custom_end()

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self):
        """
        a standard downstream evaluate process.
        """
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Test:'
        # switch to evaluation mode
        self.model.eval()

        for batch in metric_logger.log_every(self.loader_val, 10, header):
            images, target = batch[0], batch[-1]
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(self.args.use_fp16):
                output = self.model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def adjust_learning_rate(self):
        raise NotImplementedError
