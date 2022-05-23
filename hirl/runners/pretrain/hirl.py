import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import hirl.pipelines as pipelines
import hirl.utils.dist as dist_utils
from hirl import runners
from hirl.utils import misc
from hirl.utils.clustering import run_hkmeans
from hirl.utils.misc import ImageFolderInstance


def HIRLRunner(args):
    """
    runner supporting hirl pre-training. This runner dynamically inherits from a runner and build 
    hirl specific functions upon it.
    """
    base_runner_class = getattr(runners, args.pop("base_runner"))
    class DynamicHIRLRunner(base_runner_class):
        def build_dataset(self, args):
            super().build_dataset(args)
            ## "eval" dataset for extracting feature for clustering
            eval_transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

            self.dataset_val = ImageFolderInstance(os.path.join(args.data_path, 'train'), transform=eval_transform)
            eval_sampler = torch.utils.data.DistributedSampler(self.dataset_val, num_replicas=dist_utils.get_world_size(), 
                                                            rank=dist_utils.get_rank(), shuffle=False)

            self.loader_eval = torch.utils.data.DataLoader(
                        self.dataset_val, sampler=eval_sampler,
                        batch_size=int(args.batch_size)*4,
                        num_workers=args.num_workers,
                        pin_memory=True, drop_last=False
                    )
            
            self.features = None
            self.cluster_result = None

        def build_model(self, args):
            super().build_model(args)
            pipeline_class = args.pipeline.pop("name")
            self.model = getattr(pipelines, pipeline_class)(model=self.model_without_ddp.cpu(), **args.pipeline)
            self.model = self.model.cuda()
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu], find_unused_parameters=True)
            self.model_without_ddp = self.model.module
            self.model_without_ddp.T = self.args.T

        def build_optimizer(self, args):
            super().build_optimizer(args, self.model_without_ddp.model)
            ## add param groups
            added_param_groups = {"params": self.model_without_ddp.projections.parameters()}
            self.optimizer.add_param_group(added_param_groups)

        @torch.no_grad()
        def compute_feature(self):
            """
            extract feature from the training dataset with 
            eval augmentation
            """
            self.model_without_ddp.eval()
            metric_logger = misc.MetricLogger(delimiter="  ")
            header = 'Feature Extraction'
            log_interval = self.args.get("log_interval", 10)

            features = torch.zeros(len(self.dataset_val), self.model_without_ddp.input_dim).cuda()

            for it, (batch) in enumerate(metric_logger.log_every(self.loader_eval, log_interval, header)):
                images = batch['images']
                index = batch['index']
                images = images.cuda(non_blocking=True)
                feat = self.model_without_ddp.forward_feature(images)
                feat = F.normalize(feat, dim=-1)
                features[index] = feat
            dist.barrier()
            dist.all_reduce(features, op=dist.ReduceOp.SUM)
            return features.cpu()

        def run_clustering(self, features):
            self.args.num_cluster = self.args.pipeline.num_cluster

            cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster': [], 'logits': []}
            for i, num_cluster in enumerate(self.args.num_cluster):
                cluster_result['im2cluster'].append(torch.zeros(len(self.dataset_val),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), features.shape[-1]).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
                if i < (len(self.args.num_cluster) - 1):
                    cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
                    cluster_result['logits'].append(torch.zeros([int(num_cluster), int(self.args.num_cluster[i+1])]).cuda())
            if dist.get_rank() == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 
                features = features.numpy()
                cluster_result = run_hkmeans(features, self.args.num_cluster, self.args.T, self.args.local_rank) 

            dist.barrier()
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:
                    dist.broadcast(data_tensor, 0, async_op=False)

            return cluster_result

        def run(self):
            """
            perform clustering before each training epoch after the 
            cluster warmup epoch
            """
            start_time = time.time()
            print(f"Start training for {self.args.epochs} epochs")
            self.model.train()
            for epoch in range(self.start_epoch, self.args.epochs):
                self.set_epoch(epoch)
                if epoch >= self.args.cluster_warmup_epoch:
                    ## extract_feature & run clustering
                    s_time = time.time()
                    self.features = self.compute_feature()
                    e_time = time.time()
                    print("Feature extraction takes {}".format(str(datetime.timedelta(seconds=int(e_time - s_time)))))

                    s_time = time.time()
                    self.cluster_result = self.run_clustering(self.features)
                    e_time = time.time()
                    print("Clustering takes {}".format(str(datetime.timedelta(seconds=int(e_time - s_time)))))

                    ## save and upload to hdfs
                    if dist_utils.get_rank() == 0:
                        torch.save(self.cluster_result, os.path.join(self.args.output_dir, "cluster_{}.pth".format(epoch)))
                        
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

        def forward_batch(self, batch, **kwargs):
            cluster_result = self.cluster_result
            output_dict = dict()

            ## iter-wise cluster assignment
            iter_wise_assignment = self.args.get("iter_wise_assignment", False)
            if iter_wise_assignment and cluster_result is not None:
                with torch.no_grad():
                    images = batch['images']
                    assert isinstance(images, list) and len(images) > 1
                    index = batch['index']
                    key_imgs = images[1]
                    key = self.model_without_ddp.forward_feature(key_imgs.cuda())
                    im2cluster = torch.argmax(torch.mm(key, cluster_result['centroids'][0].t()), dim=-1)
                    same_rate = (im2cluster==cluster_result['im2cluster'][0][index]).float().mean().item()
                    output_dict['assignment_same_ratio'] = same_rate
                    cluster_result['im2cluster'][0][index] = im2cluster
                    for h_id in range(1, len(cluster_result['im2cluster'])):
                        assignment = cluster_result['im2cluster'][h_id-1][index]
                        cluster2cluster = cluster_result['cluster2cluster'][h_id-1]
                        cluster_result['im2cluster'][h_id][index] = cluster2cluster[assignment]
            output = super().forward_batch(batch, cluster_result=cluster_result,
                                        index=batch["index"], **kwargs)
            output_dict.update(output)
            return output_dict

        def log_metrics(self, metric_logger, output_dict):
            ## base model metric logging
            super().log_metrics(metric_logger, output_dict)
            ## log image loss
            metric_logger.update(**{"image_loss": output_dict["image_loss"]})
            ## hirl specific logging: semantic loss | semantic accuracy
            if "semantic_loss" in output_dict:
                metric_logger.update(**{"semantic_losses":output_dict["semantic_loss"]})

    return DynamicHIRLRunner(args)
