import subprocess
import argparse
import os, sys
import time

def init_workdir():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)

def parse_args():
    parser = argparse.ArgumentParser("Object Detection Eval")
    parser.add_argument("--data", type=str, default="coco", choices=["coco"])
    parser.add_argument("--pretrained", type=str, default="./")
    parser.add_argument("--backbone_prefix", type=str, default="backbone")
    parser.add_argument("--output_dir", type=str, default="./downstream/object_detection/coco/mocov2_rn50_test")

    return parser.parse_known_args()

if __name__ == "__main__":
    args, other_args = parse_args()
    init_workdir()
    ## set detectron2 default root
    os.environ["DETECTRON2_DATASETS"] = "../../datasets/"
    print("current work dir: {}".format(os.getcwd()))
    print("getting model")
    
    subprocess.call("python3 convert-pretrain-to-detectron2.py {} output.pkl --backbone_prefix {}".format(args.pretrained, args.backbone_prefix), shell=True)
    if args.data == "coco":
        config_file = "configs/coco_R_50_C4_2x_hirl.yaml"
    else:
        raise ValueError
    os.makedirs(args.output_dir, exist_ok=True)
    print("other args: {}".format(other_args))

    cmd = "python3 train_net.py --config-file {} --num-gpus 8 MODEL.WEIGHTS ./output.pkl".format(config_file)
    cmd += f" OUTPUT_DIR {args.output_dir}"
    
    for argv in other_args:
        cmd += f" {argv}"
    print("launch training")
    print(f"cmd: {cmd}")
    
    with open(os.path.join(args.output_dir, "log_train.txt"), "wb") as f:
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