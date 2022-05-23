import argparse
import os
import sys
from hirl.utils.dist import *
import hirl.backbones.vision_transformer as vt
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet
from hirl.utils import misc


class DatasetBuilder(object):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def build_imagenet(self, args):
        data_path = args.data
        dataset_train = ReturnIndexDataset(os.path.join(data_path, "train"), transform=self.transform)
        dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"), transform=self.transform)
        return dataset_train, dataset_val

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, label = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

def extract_feature_pipeline(args):

    data_builder = DatasetBuilder()
    dataset_train, dataset_val = getattr(data_builder, "build_imagenet")(args)
    # ============ preparing data ... ============
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

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

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    # by default, l2 normalization would be applied
    if dist.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels

def forward_single_vit(samples, model, n_last_blocks=1):
    """
    modified forward function for vit architecture.
    """
    intermediate_output = model.get_intermediate_layers(samples, n_last_blocks)
    output = [x[:, 0] for x in intermediate_output] # [CLS] Token for ViT
    
    feats = torch.cat(output, dim=-1).clone()
    return feats

def is_vit(model):
    if isinstance(model, vt.VisionTransformer):
        return True
    else:
        return False

@torch.no_grad()
def extract_features(model, data_loader, n_last_blocks=1, use_cuda=True, multiscale=False):
    """
    Notes:
    - n_last_blocks and avg_pool would be omiited if not using vit.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        
        if is_vit(model): # different embedding scheme for vit
            if multiscale:
                v = None
                for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
                    if s == 1:
                        inp = samples.clone()
                    else:
                        inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
                    feats = forward_single_vit(inp)
                    if v is None:
                        v = feats
                    else:
                        v += feats
                v /= 3
                v /= v.norm()
                feats = v
            else:
                feats = forward_single_vit(samples, model, n_last_blocks)
        else:
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features,
                  test_labels, k, T, num_classes=1000,):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

def parse_args_knn():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. `n=1` all the time for k-NN evaluation is used in iBOT/DINO/MoCoV3.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=1, type=int,
        help="Store features in GPU.")
    parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
    parser.add_argument("--checkpoint_key", default="state_dict", type=str,
        help='Key to use in the checkpoint')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; """)
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data', type=str, default="./datasets/Imagenet1K/ILSVRC/Data/CLS-LOC")
    parser.add_argument("--backbone_prefix", type=str, default="backbone")
    parser.add_argument("--model_prefix", type=str, default="model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_knn()
    init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()
