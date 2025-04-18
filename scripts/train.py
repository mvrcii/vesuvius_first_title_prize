import argparse
import os
import types
import warnings
from datetime import datetime

import numpy as np
import torch
import wandb
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning_fabric.accelerators import find_usable_cuda_devices

from phoenix.models.data_modules.segformer_datamodule import SegFormerDataModule
from phoenix.models.data_modules.unetrsf_datamodule import UNETR_SF_DataModule
from phoenix.models.lightning_modules.segformer_module import SegformerModule
from phoenix.models.lightning_modules.unetrsf_module import UNETR_SF_Module
from phoenix.utility.configs import Config

warnings.filterwarnings("ignore", message="Some weights * were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task")
warnings.simplefilter("ignore", category=Warning)

torch.set_float32_matmul_precision('medium')


def log_wandb_hyperparams(config, wandb_logger):
    config_dict = vars(config)
    cleaned_config = {k: v for k, v in config_dict.items() if not isinstance(v, types.ModuleType)}
    wandb_logger.log_hyperparams(cleaned_config)


def get_callbacks(cfg, model_run_dir):
    node = getattr(cfg, 'node', True)

    # Only save model checkpoint if we are on a gpu node
    if node:
        os.makedirs(model_run_dir, exist_ok=True)
        cfg.save_to_file(model_run_dir)
        monitor_metric = getattr(cfg, "monitor_metric", "val_iou")
        filename_metric = f"{{{monitor_metric}:.2f}}"
        print("Monitoring metric:", monitor_metric)

        # Determine mode based on metric name
        mode = "min" if "loss" in monitor_metric else "max"

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_run_dir,
            filename=f"best-checkpoint-{{epoch}}-{filename_metric}",
            save_top_k=3,
            monitor=monitor_metric,
            mode=mode,
            every_n_epochs=1
        )

        return [checkpoint_callback]
    else:
        return []


def get_model_run_name(config, logger):
    # Model name
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = getattr(config, 'model_name', 'default_model')
    wandb_generated_name = logger.experiment.name
    model_run_name = f"{wandb_generated_name}-{model_name}-{timestamp}"
    logger.experiment.name = model_run_name
    model_run_dir = os.path.join(config.work_dir, "checkpoints", model_run_name)
    return model_run_name, model_run_dir


def get_device_configuration(gpu):
    """
    Determines the appropriate device configuration for training based on
    the availability of CUDA-enabled GPUs.

    :return: A tuple (accelerator, devices) where:
        - 'accelerator' is a string indicating the type of accelerator ('gpu' or 'cpu').
        - 'devices' is an int or list indicating the devices to be used.
    """
    if gpu != 0:
        return [gpu]
    else:
        if torch.cuda.is_available():
            # Return all available GPUs
            gpu_ids = find_usable_cuda_devices()
            return gpu_ids
        else:
            # No GPUs available, use CPU
            return 1


def check_wandb_login():
    if not wandb.api.api_key:
        print("You are not logged in to Weights and Biases. Please log in using `wandb.login()`.")
        # Optionally, you could raise an exception to stop the script here
        raise ValueError("wandb is not logged in.")
    else:
        print("You are logged into Weights and Biases.")


def main(config_path, seed=42, gpu=0, checkpoint=None, resume_with_reload=False):
    config = Config.load_from_file(config_path)

    seed = config.seed if seed is None else seed
    seed_everything(seed)
    np.random.seed(seed)

    check_wandb_login()
    wandb_logger = WandbLogger(project="phoenix", entity="wuesuv")
    log_wandb_hyperparams(config, wandb_logger=wandb_logger)

    model_run_name, model_run_dir = get_model_run_name(config, wandb_logger)

    if checkpoint:
        model_run_name = f"{model_run_name}-finetune"
        wandb_logger.experiment.name = model_run_name
        scroll_id = getattr(config, 'scroll_id', '5')
        model_run_dir = os.path.join(config.work_dir, "checkpoints", f'scroll{str(scroll_id)}', model_run_name)
        print(f"Fine-tuning from checkpoint: {checkpoint}")

    config_dict = config.to_clean_dict()

    if config.architecture == "segformer":
        model = SegformerModule(**config_dict)
        data_module = SegFormerDataModule(cfg=config)
    else:
        model = UNETR_SF_Module(**config_dict)
        if resume_with_reload:
            model.load_state_dict(torch.load(checkpoint)["state_dict"])
        data_module = UNETR_SF_DataModule(cfg=config)

    callbacks = get_callbacks(config, model_run_dir=model_run_dir)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        devices=get_device_configuration(gpu),
        enable_progress_bar=True,
        precision="16-mixed",
        check_val_every_n_epoch=config.val_interval,
        callbacks=callbacks,
        gradient_clip_val=config.gradient_clip_val,
    )

    devices_used = get_device_configuration(gpu)
    print(f"Training on devices: {devices_used}")

    trainer.fit(model, data_module, ckpt_path=checkpoint if not resume_with_reload else None)


def parse_args():
    parser = argparse.ArgumentParser(description='Train configuration.')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for the script')
    parser.add_argument('--gpu', type=int, default=0, help='Cuda GPU (default: 0)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for fine-tuning')
    parser.add_argument('--resume-with-reload', action='store_true', help='Resume training with reloading fresh optimizer and scheduler')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # We need a checkpoint given if we call resume_with_reload
    assert args.resume_with_reload == False or args.checkpoint

    main(args.config_path, args.seed, args.gpu, args.checkpoint, args.resume_with_reload)
