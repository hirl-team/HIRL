import argparse
import os, sys
import yaml
from hirl import runners
import ast
from addict import Dict as adict
from hirl.utils.config_utils import update_config

def parse_args():
    parser = argparse.ArgumentParser("Running pretrain scheme")
    parser.add_argument("--config", "-c", type=str, 
                        help="which config file to use")
    parser.add_argument("--local_rank", type=int, default=-1)
    args, other_args = parser.parse_known_args()
    return args, other_args

if __name__ == "__main__":
    args, other_args = parse_args()
    with open(args.config, 'r') as c:
        config = yaml.safe_load(c)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    
    ## update configs with args
    config = adict(config)
    config.local_rank = args.local_rank
    update_config(config, other_args)

    runner_class = config.get("runner")

    runner = getattr(runners, runner_class)(config)
    runner.run()