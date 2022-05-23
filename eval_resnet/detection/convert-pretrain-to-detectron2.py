#!/usr/bin/env python
"""
adopted from https://github.com/facebookresearch/moco/tree/master/detection
"""
import pickle as pkl
import sys
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser("convert pretrain to detectron2 pkl.")
    parser.add_argument("input_file", type=str, default="./out.pth")
    parser.add_argument("output_file", type=str, default="./output.pkl")
    parser.add_argument("--backbone_prefix", type=str, default="backbone")
    parser.add_argument("--model_prefix", type=str, default="model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    obj = torch.load(args.input_file, map_location="cpu")
    obj = obj[args.model_prefix]

    newmodel = {}
    for k, v in obj.items():
        if not (k.startswith(f"module.{args.backbone_prefix}") or k.startswith(args.backbone_prefix)):
            continue
        old_k = k
        if k.startswith("module"):
            k = k.replace("module.", "")
        k = k.replace(args.backbone_prefix+".", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "HIRL", "matching_heuristics": True}
    for k, v in res['model'].items():
        print(k)

    with open(args.output_file, "wb") as f:
        pkl.dump(res, f)
