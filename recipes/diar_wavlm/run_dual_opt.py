# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# Copyright 2025 Nanjing University (author: Zeyan Song, clovermaxszy@gmail.com)

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from SDtrainZen.logger import init_logging_logger
from SDtrainZen.utils import instantiate
from SDtrainZen.ckpt_utils import average_ckpt


def run(config, resume):
    init_logging_logger(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config["meta"]["mixed_precision"],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        accelerator.print('fine-tuning...')
        model = average_ckpt(config["finetune"]["ckpt_dir"], model, val_metric='Loss',
                             avg_ckpt_num=config["finetune"]["avg_ckpt_num"], val_mode=config["finetune"]["val_mode"])
        print('checkpoint loaded')

    optimizer_small = instantiate(
        config["optimizer_small"]["path"],
        args={
            "params": model.small_lr_parameters(),
            "lr": config["optimizer_small"]["args"]["lr"],
            "betas": config["optimizer_small"]["args"]["betas"],
            "eps": config["optimizer_small"]["args"]["eps"],
            "weight_decay": config["optimizer_small"]["args"]["weight_decay"]
        }
    )
    optimizer_big = instantiate(
        config["optimizer_big"]["path"],
        args={
            "params": model.big_lr_parameters(),
            "lr": config["optimizer_big"]["args"]["lr"],
            "betas": config["optimizer_big"]["args"]["betas"],
            "eps": config["optimizer_big"]["args"]["eps"],
            "weight_decay": config["optimizer_big"]["args"]["weight_decay"]
        }
    )

    (model, optimizer_small, optimizer_big) = accelerator.prepare(model, optimizer_small, optimizer_big)

    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step
    if "train" in args.mode:
        train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)
        train_dataloader = DataLoader(
            dataset=train_dataset, collate_fn=train_dataset.collater, shuffle=True, **config["train_dataset"]["dataloader"]
        )
        train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
        validate_dataloader = DataLoader(
            dataset=validate_dataset, collate_fn=validate_dataset.collater, shuffle=True, **config["validate_dataset"]["dataloader"]
        )
        validate_dataloader = accelerator.prepare(validate_dataloader)

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model=model,
        optimizer_small=optimizer_small,
        optimizer_big=optimizer_big
    )

    for flag in args.mode:
        if flag == "train":
            trainer.train(train_dataloader, validate_dataloader)
        elif flag == "validate":
            trainer.validate(validate_dataloader)
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN based TS-VAD framework")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.yaml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = OmegaConf.load(config_path.as_posix())

    if "exp_id" not in config["meta"]:
        config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    run(config, args.resume)
