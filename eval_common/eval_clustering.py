import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from hirl.utils import misc
from hirl.utils.clustering import get_y_preds, run_hkmeans
from hirl.utils.dist import *
from sklearn import metrics
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, label = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

def eval_pred(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ami = metrics.adjusted_mutual_info_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ami, acc

def run_clustering(features, args):
    """
    eval clustering result by kmeans clustering with features and labels.

    Args:
        features: torch.tensor([N, D])
        labels: torch.tensor([N,])
    Returns:
        dict(acc, nmi, ami)
    """
    num_clusters = [int(x) for x in args.num_classes.split(",")]

    cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster': [], 'logits': []}
    for i, num_cluster in enumerate(num_clusters):
        cluster_result['im2cluster'].append(torch.zeros(len(features),dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(int(num_cluster), features.shape[-1]).cuda())
        cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
        if i < (len(num_clusters) - 1):
            cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
            cluster_result['logits'].append(torch.zeros([int(num_cluster), int(num_clusters[i+1])]).cuda())

    if dist.get_rank() == 0:
        features[torch.norm(features,dim=1)>1.5] /= 2 
        features = features.cpu().numpy()
        cluster_result = run_hkmeans(features, num_clusters) 

    dist.barrier()
    for k, data_list in cluster_result.items():
        for data_tensor in data_list:
            dist.broadcast(data_tensor, 0, async_op=False)
    
    im2cluster = cluster_result['im2cluster'][-1].cpu().long() # [N,]
    print("number of unique clusters: {}".format(len(im2cluster.unique())))
    print("max cluster id {}".format(im2cluster.max()))

    return im2cluster


@torch.no_grad()
def extract_features(model, loader, args):
    model.eval()
    header = 'Feature Extraction'
    log_interval = 100
    metric_logger = misc.MetricLogger(delimiter="  ")
    ## lazy feat dim
    image = next(iter(loader))[0].cuda()
    feat = model(image)
    feat_dim = feat.shape[-1]

    features =  torch.zeros(len(loader.dataset), feat_dim).cuda()
    print("feature shape: {}".format(features.shape))

    for it, (image, index) in enumerate(metric_logger.log_every(loader, log_interval, header)):
        image = image.cuda(non_blocking=True)
        feat = F.normalize(model(image), dim=-1)
        features[index] = feat.detach()
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    features[torch.norm(features,dim=1)>1.5] /= 2 
    return features.cpu()

def run(args):
    # ============ preparing data ... ============
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_val = ReturnIndexDataset(os.path.join(args.data, "val"), transform=transform)
    test_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=10,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    if args.arch in resnet.__dict__.keys():
        model = resnet.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    elif args.arch in vt.__dict__.keys():
        model = vt.__dict__[args.arch](
            patch_size=args.patch_size, 
            num_classes=0,
            use_mean_pooling=False)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()

    def load_pretrained_weights(model, pretrained, backbone_prefix=None, model_prefix="model", filtered_keys=[]):
        checkpoint = torch.load(pretrained, map_location="cpu")
        if len(model_prefix):
            checkpoint_model = checkpoint[model_prefix]
        else:
            checkpoint_model = checkpoint
        ## automatically remove ddp prefix
        if all([k.startswith("module.") for k in checkpoint_model.keys()]):
            print("remove ddp prefix from model.")
            checkpoint_model = {k.replace("module.", ""):v for k,v in checkpoint_model.items()}
        
        if backbone_prefix:
            checkpoint_model = {k[len(backbone_prefix)+1:]:v for k,v in checkpoint_model.items() if k.startswith(backbone_prefix)}
        
        state_dict = model.state_dict()
        ## remove head / fc
        removed_keys = list()
        for key in checkpoint_model.keys():
            if key not in state_dict or key in filtered_keys or checkpoint_model[key].shape != state_dict[key].shape:
                removed_keys.append(key)

        print("removed keys in pretrained model: {}".format(removed_keys))
        for key in removed_keys:
            checkpoint_model.pop(key)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("loading message: {}".format(msg))
        return msg

    load_pretrained_weights(model, args.pretrained, args.backbone_prefix, args.model_prefix)
    model.eval()

    # ============ extract features... ============
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    test_features = extract_features(model, data_loader_val, args)

    im2cluster = run_clustering(test_features, args)
    nmi, ami, acc = eval_pred(test_labels.cpu().numpy(), im2cluster.cpu().numpy())
    return_dict = dict(nmi=nmi, ami=ami, acc=acc)
    return return_dict

def parse_args():
    parser = argparse.ArgumentParser('Evaluation with kmeans clustering on ImageNet')
    parser.add_argument("--arch", type=str, default="resnet50", help="Architecture of network.")
    parser.add_argument("--backbone_prefix", type=str, default="backbone")
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--num_classes", type=str, default="1000")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument('--data', type=str, default="./datasets/Imagenet1K/ILSVRC/Data/CLS-LOC")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; """)
    args, other_args = parser.parse_known_args()
    return args, other_args

if __name__ == '__main__':
    args, other_args = parse_args()
    init_distributed_mode(args)
    cudnn.benchmark = True
    val_results = run(args)
    
    val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
    if get_rank() == 0:
        # print("result on train: {}".format(train_str))
        print("result on val: {}".format(val_str))
    dist.barrier()
