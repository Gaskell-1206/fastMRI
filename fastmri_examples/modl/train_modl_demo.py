"""
Demo script for training/testing MoDL model using fastMRI data pipeline.
"""
import os
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule
from fastmri.pl_modules.modl_module import MoDLModule

def cli_main(args):
    pl.seed_everything(args.seed)

    # Data transforms
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler="ddp",
    )

    # Model
    model = MoDLModule(
        n_layers=args.n_layers,
        k_iters=args.k_iters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        deterministic=args.deterministic,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=args.callbacks if hasattr(args, 'callbacks') else None,
    )

    # Run
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

def build_args():
    parser = ArgumentParser()
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    num_gpus = 2 if backend == "ddp" else 1
    batch_size = 1
    from fastmri.data.mri_data import fetch_dir
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = pathlib.Path(fetch_dir("log_path", path_config)) / "modl" / "modl_demo"
    
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Set deterministic training')
    parser.add_argument('--default_root_dir', type=str, default=str(default_root_dir), help='Path to default root directory')

    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="equispaced_fraction",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        mask_type="equispaced_fraction",
        challenge="multicoil",
        batch_size=batch_size,
        test_path=None,
    )

    parser = MoDLModule.add_model_specific_args(parser)
    parser.set_defaults(
        n_layers=5,
        k_iters=10,
        lr=1e-3,
        weight_decay=0.0,
    )

    args = parser.parse_args()

    # Checkpointing
    checkpoint_dir = pathlib.Path(args.default_root_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
    ]
    if getattr(args, "resume_from_checkpoint", None) is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
    return args

def run_cli():
    args = build_args()
    cli_main(args)

if __name__ == "__main__":
    run_cli()
