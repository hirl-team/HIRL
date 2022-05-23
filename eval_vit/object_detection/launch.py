#!/usr/bin/python3
import os
import sys
import socket
import random
import argparse
import subprocess
import time
import torch

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def _get_rand_port():
    return random.randrange(20000, 60000)

def init_workdir():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vit detection launcher')
    parser.add_argument('--np', type=int, default=-1,
                        help='number of processes per node')
    parser.add_argument('--nn', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--port', type=int, default=-1,
                        help='master port for communication')
    parser.add_argument('--nr', type=int, default=0, 
                        help='node rank.')
    parser.add_argument('--master_address', '-ma', type=str, default="127.0.0.1")
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--arch', type=str, default='vit_base')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--backbone_prefix', type=str, default='backbone')


    args, other_args = parser.parse_known_args()
    init_workdir()
    master_address = args.master_address
    num_processes_per_worker = torch.cuda.device_count() if args.np < 0 else args.np
    num_workers = args.nn
    node_rank = args.nr

    if args.port > 0:
        master_port = args.port
    elif num_workers == 1:
        master_port = _find_free_port()
    else: 
        master_port = _get_rand_port()

    os.makedirs(args.output_dir, exist_ok=True)
    # extract backbone weight
    if os.path.exists("out_converted.pth"):
        subprocess.call("rm out_converted.pth", shell=True)
    subprocess.call("python3 ./extract_backbone_weight.py {} out_converted.pth --backbone_prefix {}".format(args.pretrained, args.backbone_prefix), shell=True)

    # start training
    print(f'Start training on coco by torch.distributed.launch with port {master_port}!', flush=True)
    os.environ['NPROC_PER_NODE'] = str(num_processes_per_worker)
    cmd = f'python3 -m torch.distributed.launch \
            --nproc_per_node={num_processes_per_worker} \
            --nnodes={num_workers} \
            --node_rank={node_rank} \
            --master_addr={master_address} \
            --master_port={master_port} \
            train.py \
            --launcher pytorch \
            ./configs/cascade_rcnn/{args.arch}_giou_4conv1f_coco_3x.py \
            --work-dir {args.output_dir} \
            --deterministic \
            --cfg-options model.backbone.use_checkpoint=True \
            model.pretrained=out_converted.pth'
    for argv in other_args:
        cmd += f' {argv}'

    with open(os.path.join(args.output_dir, 'coco_train.txt'), 'wb') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            text = proc.stdout.readline()
            f.write(text)
            f.flush()
            sys.stdout.buffer.write(text)
            sys.stdout.buffer.flush()
            exit_code = proc.poll()
            if exit_code is not None:
                break

    # start testing
    print(f'Start testing on coco by torch.distributed.launch with port {master_port}!', flush=True)
    cmd = f'python3 -m torch.distributed.launch \
            --nproc_per_node={num_processes_per_worker} \
            --nnodes={num_workers} \
            --node_rank={node_rank} \
            --master_addr={master_address} \
            --master_port={master_port} \
            ./test.py \
            --launcher pytorch \
            ./configs/cascade_rcnn/{args.arch}_giou_4conv1f_coco_3x.py \
            {os.path.join(args.output_dir, "latest.pth")} \
            --eval bbox segm \
            --cfg-options model.backbone.use_checkpoint=True'

    for argv in other_args:
        cmd += f' {argv}'

    with open(os.path.join(args.output_dir, 'log_coco_test.txt'), 'wb') as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            text = proc.stdout.readline()
            f.write(text)
            f.flush()
            sys.stdout.buffer.write(text)
            sys.stdout.buffer.flush()
            exit_code = proc.poll()
            if exit_code is not None:
                break

    sys.exit(exit_code)
