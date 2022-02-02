"""Data loader."""


import torch

from torch.utils.data.sampler import RandomSampler
from .kinetics import Kinetics

_DATASETS = {
    "kinetics": Kinetics,
}


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            vml/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val"]
    dataset_name = cfg.TRAIN.DATASET
    batch_size = cfg.TRAIN.BATCH_SIZE

    if split in ["train"]:
        shuffle = True

    elif split in ["val"]:
        shuffle = False

    # Construct the dataset
    dataset = _DATASETS[dataset_name](cfg, split)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader

