import os
import sys
import toml
import argparse
from argparse import ArgumentParser
from os.path import join


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.bsrnn_svs_model import BSRNNSVSModel, get_argparse_groups
from models.data_module import SpecsDataModule
from models.sgmse.util.other import set_torch_cuda_arch_list

torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision("high")


def merge_config_into_argv(raw_argv, config_dict):
    argv = list(raw_argv)
    cli_args = {token for token in argv if token.startswith("--")}

    for key, value in config_dict.items():
        arg_name = f"--{key.replace('_', '-')}"
        if arg_name in cli_args:
            continue

        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
        elif isinstance(value, list):
            argv.extend([arg_name] + [str(v) for v in value])
        else:
            argv.extend([arg_name, str(value)])

    return argv


def load_toml_config(config_path):
    return toml.load(config_path)


if __name__ == "__main__":
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()

    for p in (base_parser, parser):
        p.add_argument("--config", type=str, default=None, help="Path to TOML config file.")
        p.add_argument("--nolog", action="store_true", help="Disable wandb logging.")
        p.add_argument("--audio-log-interval", type=int, default=5)
        p.add_argument("--wandb-name", type=str, default=None)
        p.add_argument("--ckpt", type=str, default=None, help="Trainer resume checkpoint.")
        p.add_argument("--log-dir", type=str, default="logs")
        p.add_argument("--save-ckpt-interval", type=int, default=50000)
        p.add_argument("--start-with-validation", action="store_true")
        p.add_argument("--run-id", type=str, default=None)

    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu")
    trainer_parser.add_argument("--devices", default="auto")
    trainer_parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    trainer_parser.add_argument("--max-epochs", type=int, default=550)

    BSRNNSVSModel.add_argparse_args(
        parser.add_argument_group("BSRNNSVSModel", description=BSRNNSVSModel.__name__)
    )
    SpecsDataModule.add_argparse_args(
        parser.add_argument_group("DataModule", description=SpecsDataModule.__name__)
    )

    temp_args, _ = base_parser.parse_known_args()
    config_dict = {}
    if temp_args.config:
        config_dict = load_toml_config(temp_args.config)
        print(f"Loaded config from {temp_args.config}")

    argv_to_parse = merge_config_into_argv(sys.argv[1:], config_dict)
    args = parser.parse_args(argv_to_parse)
    arg_groups = get_argparse_groups(parser, args)

    if not args.nolog:
        wandb.login()

    model = BSRNNSVSModel(
        data_module_cls=SpecsDataModule,
        **{
            **vars(arg_groups["BSRNNSVSModel"]),
            **vars(arg_groups["DataModule"]),
            "audio_log_interval": args.audio_log_interval,
            "nolog": args.nolog,
        },
    )

    logger = None
    if not args.nolog:
        logger = WandbLogger(
            project="finetuning-for-svs",
            log_model=False,
            save_dir=args.log_dir,
            name=args.wandb_name,
            id=args.run_id,
            resume="allow",
        )
        logger.experiment.log_code(".")

    callbacks = []
    if logger is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_last=True,
                filename="{epoch}-last",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}"),
                filename="{step}",
                save_top_k=-1,
                every_n_train_steps=args.save_ckpt_interval,
            )
        )
        # Track all four metrics as in finetune_bsrnnse.py
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="sdr",
                mode="max",
                filename="{epoch}-{sdr:.2f}",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="si_sdr",
                mode="max",
                filename="{epoch}-{si_sdr:.2f}",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="multi_res_loss",
                mode="min",
                filename="{epoch}-{multi_res_loss:.2f}",
            )
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=1,
                monitor="mert_mse",
                mode="min",
                filename="{epoch}-{mert_mse:.4f}",
            )
        )

    trainer = pl.Trainer(
        **vars(arg_groups["Trainer"]),
        strategy="ddp",
        logger=logger,
        callbacks=callbacks if callbacks else None,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    model.data_module.setup(stage="fit")

    if args.start_with_validation:
        model.eval()
        trainer.validate(model, ckpt_path=args.ckpt)

    trainer.fit(model, ckpt_path=args.ckpt)
