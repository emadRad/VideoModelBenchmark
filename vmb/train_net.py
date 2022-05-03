#!/usr/bin/env python3

"""Train a video classification model."""
import os

import numpy as np
import pprint
import torch
from tqdm import tqdm

import vmb.models.losses as losses
import vmb.models.optimizer as optim
import vmb.utils.checkpoint as cu
import vmb.utils.metrics as metrics
import vmb.utils.misc as misc
from vmb.datasets import loader
from vmb.models import model_builder
from vmb.utils.meters import TrainMeter, ValMeter
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

import vmb.utils.logging as logging

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, device
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
        device (torch.device): training device gpu or cpu.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    lr = optim.get_epoch_lr(optimizer)
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with record_function("model_train"):
            for cur_iter, (inputs, labels) in tqdm(
                enumerate(train_loader), total=data_size
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Perform the forward pass.
                preds = model(inputs)

                # Compute the loss.
                loss = loss_fun(preds, labels)

                # check Nan Loss.
                misc.check_nan_losses(loss, model, optimizer, cfg, cur_epoch, cur_iter)

                loss.backward()

                # Update the parameters.
                optimizer.step()
                # Zero grad.
                optimizer.zero_grad()

                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
                train_meter.iter_toc()

                # Update and log stats.
                train_meter.update_stats(top1_err, top5_err, loss, lr, inputs[0].size(0))

                train_meter.log_stats_tensorboard()
                train_meter.log_iter_stats(cur_epoch, cur_iter)
                train_meter.iter_tic()

            # Log epoch stats.
            train_meter.log_epoch_stats(cur_epoch)
            train_meter.reset()
    epoch_total_sec = misc.format_time_in_sec(prof.key_averages().total_average().cuda_time_total)
    logger.info(f"Epoch {cur_epoch} took {epoch_total_sec} seconds on GPU(s).")
    return epoch_total_sec


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, device):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        device (torch.device):
    """
    model.eval()
    val_meter.iter_tic()
    logger.info(f"Validation Started")

    for cur_iter, (inputs, labels) in tqdm(
        enumerate(val_loader), total=len(val_loader)
    ):
        # Transfer the data to the current GPU device.
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(top1_err, top5_err, inputs[0].size(0))

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    top1_acc = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return top1_acc


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            sgs/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)

    if cfg.NUM_GPUS > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to device(s)
    model = model.to(device)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    scheduler = optim.construct_lr_scheduler(cfg, optimizer)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(
        cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
    ):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(
            cfg.OUTPUT_DIR, cfg.TRAIN.RESUME_EXPR_NUM
        )
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint,
            model=model,
            fine_tune=False,
            num_gpus=cfg.NUM_GPUS,
            optimizer=optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.TRAIN.FINE_TUNE,
            cfg.NUM_GPUS,
            optimizer=optimizer,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0
        logger.info("Training from scratch")

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create tensorboard summary writer
    writer = SummaryWriter(cfg.SUMMARY_PATH)
    writer.add_text(f"Config_EXPR_NUM={cfg.EXPR_NUM}", pprint.pformat(cfg).replace("\n", "\n\n"))

    # Create meters.
    train_meter = TrainMeter(
        len(train_loader), cfg, start_epoch * (len(train_loader)), writer
    )
    val_meter = ValMeter(len(val_loader), cfg, writer)

    # Print summary path.
    logger.info("Summary path {}".format(cfg.SUMMARY_PATH))

    logger.info("Process PID {}".format(os.getpid()))
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    sum_epoch_times = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        epoch_time_sec = train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, device
        )
        sum_epoch_times += epoch_time_sec
        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            logger.info("Saving checkpoint")
            cu.save_checkpoint(
                cfg.OUTPUT_DIR, cfg.EXPR_NUM, model, optimizer, cur_epoch, cfg
            )

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            top1_acc = eval_epoch(val_loader, model, val_meter, cur_epoch, device)

        scheduler.step()
    writer.flush()
    avg_epoch_sec = sum_epoch_times / cfg.SOLVER.MAX_EPOCH
    return top1_acc, avg_epoch_sec
