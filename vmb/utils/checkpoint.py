"""Functions that handle saving and loading of checkpoints."""

import os
from collections import OrderedDict
import torch

import vmb.utils.logging as logging

logger = logging.get_logger(__name__)


def get_checkpoint_in_path(path):
    names = os.listdir(path) if os.path.exists(path) else []
    names = [f for f in names if f.endswith(".pyth")]
    assert len(names), "No checkpoints found in '{}'.".format(path)
    name = sorted(names)[-1]
    return os.path.join(path, name)


def make_checkpoint_dir(path_to_job, expr_num):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints", str(expr_num))
    # Create the checkpoint dir from the master process
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job, resume_expr_num):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints", str(resume_expr_num))


def get_path_to_checkpoint(path_to_job, epoch, expr_num):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return os.path.join(get_checkpoint_dir(path_to_job, expr_num), name)


def get_last_checkpoint(path_to_job, resume_expr_num):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    assert resume_expr_num != None, "No experiment number is given"
    d = get_checkpoint_dir(path_to_job, str(resume_expr_num))
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job, expr_num):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job, str(expr_num))
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        checkpoint_period (int): the frequency of checkpointing.
    """
    if checkpoint_period == -1:
        return False
    return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_job, expr_num, model, optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """

    # Ensure that the checkpoint dir exists.
    os.makedirs(get_checkpoint_dir(path_to_job, expr_num), exist_ok=True)
    saving_model = model.module if cfg.NUM_GPUS > 1 else model
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, epoch + 1, expr_num)
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            logger.info("Inflate {}: {} -> {}: {}".format(k, v2d.shape, k, v3d.shape))
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
        if v2d.shape == v3d.shape:
            v3d = v2d
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def load_checkpoint(
    path_to_checkpoint,
    model,
    fine_tune,
    num_gpus,
    optimizer=None,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        fine_tune (bool): whether it is fine-tuning or not
        num_gpus (int): number of gpus
        optimizer (optim): optimizer to load the historical state.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert os.path.exists(path_to_checkpoint), "Checkpoint '{}' not found".format(
        path_to_checkpoint
    )
    logger.info(f"Loading from {path_to_checkpoint}")
    ms = model.module if num_gpus > 1 else model
    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    ms.load_state_dict(checkpoint["model_state"])
    if not fine_tune and optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys() and not fine_tune:
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    return epoch
