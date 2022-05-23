import os

import torch
from hirl.utils import dist as dist_utils
from torchvision import datasets

from .finetune import FinetuneRunner


class SemiSupRunner(FinetuneRunner):
    """
    Imagenet fine-tune on part of labels. The core difference 
    from finetune runner is the dataset building.
    """
    filtered_keys = []
    def build_dataset(self, args):
        """
        Enable passing a label file indicating the subset to use.
        """
        transform_train = self.build_transform(True, args)
        transform_val = self.build_transform(False, args)

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), 
                                        transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), 
                                        transform=transform_val)

        ### training subset (only support for imagenet) ###
        labels_perc = args.get("labels_perc", None)
        if labels_perc:
            subset_file = "hirl/misc/{}percent.txt".format(labels_perc)
            with open(subset_file, "r") as f:
                list_imgs = f.readlines()
                list_imgs = [x.split("\n")[0] for x in list_imgs]
            train_data_path = os.path.join(args.data_path, 'train')
            dataset_train.samples = [(
                os.path.join(train_data_path, li.split('_')[0], li),
                dataset_train.class_to_idx[li.split('_')[0]]
            ) for li in list_imgs]

            print("Training with {} percent of training data".format(labels_perc))
        ###################################################
        
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
