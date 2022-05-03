import math
import numpy as np
import os
from datetime import datetime
import torch


import vmb.utils.logging as logging
from vmb.utils.checkpoint import get_checkpoint_dir

logger = logging.get_logger(__name__)


def format_time_in_sec(time_us):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    return time_us / US_IN_SECOND


def check_nan_losses(loss, model, optimizer, cfg, cur_epoch, cur_iter):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        saving_model = model.module if cfg.NUM_GPUS > 1 else model
        os.makedirs(get_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM), exist_ok=True)
        name = "nan_loss_checkpoint_epoch_{:05d}.pyth".format(cur_epoch+1)
        path_to_checkpoint = os.path.join(get_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM), name)
        checkpoint = {
            "epoch": cur_epoch,
            "iter": cur_iter,
            "optimizer_state": optimizer.state_dict(),
            "model_state": saving_model.state_dict(),
            "cfg": cfg.dump(),
        }
        torch.save(checkpoint, path_to_checkpoint)
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        cur_epoch (int): current epoch.
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def find_latest_experiment(path):
    if not os.path.exists(path):
        return 0

    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path
