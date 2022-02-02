"""Optimizer."""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class IdentityPolicy(_LRScheduler):
    def step(self):
        pass


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    optim_params = model.parameters()
    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def construct_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_POLICY == 'identity':
        return IdentityPolicy(optimizer)
    elif cfg.SOLVER.LR_POLICY == "cosine_linear_warmup":
        main_lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.SOLVER.T_MAX - cfg.SOLVER.WARMUP_EPOCHS, eta_min=0.00001,
        )
        warmup_lr_scheduler = LinearLR(
            optimizer, start_factor=cfg.SOLVER.WARMUP_FACTOR, total_iters=cfg.SOLVER.WARMUP_EPOCHS
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[cfg.SOLVER.WARMUP_EPOCHS]
                )
        return lr_scheduler
    else:
        raise NotImplementedError(f"LR scheduler {cfg.SOLVER.LR_POLICY} is not found.")


def get_epoch_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
