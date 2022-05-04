#!/usr/bin/env python3
import csv
import os
import sys
from os.path import join, split, splitext
import argparse

from fvcore.common.timer import Timer
import vmb.utils.checkpoint as cu
from vmb.config.defaults import get_cfg
import vmb.utils.misc as misc
from vmb.train_net import train

import json


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        resume_expr_num (int): the number of the experiment to resume.
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/X3D_M.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See vml/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "resume_expr_num"):
        cfg.RESUME_EXPR_NUM = args.resume_expr_num
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    timer = Timer()
    # Start to record time.
    timer.reset()

    args = parse_args()
    cfg = load_config(args)
    cfg.CONFIG_FILE = args.cfg_file
    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, cfg_file_name)

    summary_path = misc.check_path(join(cfg.OUTPUT_DIR, "summary"))
    cfg.EXPR_NUM = misc.find_latest_experiment(join(cfg.OUTPUT_DIR, "summary")) + 1
    # result folder to store  time and accuracy
    os.makedirs("results", exist_ok=True)
    result_file = f"results/{cfg_file_name}_{cfg.TRAIN.DATASET}_expr{cfg.EXPR_NUM:04d}.csv"

    if cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
        cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
    cfg.SUMMARY_PATH = misc.check_path(join(summary_path, "{}".format(cfg.EXPR_NUM)))
    cfg.CONFIG_LOG_PATH = misc.check_path(
        join(cfg.OUTPUT_DIR, "config", "{}".format(cfg.EXPR_NUM))
    )
    with open(os.path.join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)
    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    # Perform training.
    if cfg.TRAIN.ENABLE:
        val_top1_acc, avg_train_sec = train(cfg=cfg)
        timer.pause()
        result_dict = {"val_top1_acc": round(val_top1_acc, 2),
                       "num_gpus": cfg.NUM_GPUS,
                       "batch_size": cfg.TRAIN.BATCH_SIZE,
                       "num_epochs": cfg.SOLVER.MAX_EPOCH,
                       "num_workers": cfg.DATA_LOADER.NUM_WORKERS,
                       "train_time_in_sec": round(avg_train_sec, 2),
                       "elapsed_time_in_sec": round(timer.seconds(), 3)}
        with open(result_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=result_dict.keys())
            writer.writeheader()
            writer.writerow(result_dict)


if __name__ == "__main__":
    main()