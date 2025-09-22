import argparse
import datetime
import json
import os
import shutil
from glob import glob
from itertools import chain

import torch

import dnnlib
from torch_utils import dist
from training import training_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',   type=int, default=512)
    parser.add_argument('--data-class',         type=str, default="datautils.ffhq.FFHQ64", help="Pythonic dataset class")
    parser.add_argument('--ref-path',           type=str, default="dataset-refs/ffhq-64-val.pkl", help="Precomputed reference path")
    parser.add_argument('--lr',                 type=float, default=5e-5, help="Training learning rate")
    parser.add_argument('--decay',              type=float, default=15, help="Learning rate decay")
    parser.add_argument('--P_mean',             type=float, default=-0.8, help="Timestep lognormal mean")
    parser.add_argument('--P_std',              type=float, default=1.6, help="Timestep lognormal std")
    parser.add_argument('--duration',           type=int, default=25, help="Training duration")
    parser.add_argument('--degradation',        type=str, default=None, help="Pythonic degradation class")
    parser.add_argument('--augment',            type=float, default=0.0, help="Augmentation probability")
    parser.add_argument('--mse-train',          action="store_true", help="Train only MMSE predictor")
    parser.add_argument('--blind-train',        action="store_true", help="Train blind denoiser")
    parser.add_argument('--nppc-loss',          type=float, default=0.0, help="Use NPPC loss")
    parser.add_argument('--nppc-outputs',       type=int, default=1, help="Number of NPPC outputs")
    parser.add_argument('--noise-level',        type=float, default=0.0, help="Inverse problem noise sigma")
    parser.add_argument('--label-drop',         type=float, default=0.0, help="Label drop probability")

    parser.add_argument('--batch-gpu',          type=int, default=512, help="Maximum batch in gpu (grad accumulation)")
    parser.add_argument('--fp32',               action="store_true", help="Train in full precision")

    parser.add_argument('--debug',              action="store_true", help="Debug mode, no logging")
    parser.add_argument('--checkpoint-freq',    type=int, default=23)
    parser.add_argument('--status',             type=int, default=17)
    parser.add_argument('--sample-freq',        type=int, default=20)
    parser.add_argument('--seed',               type=int, default=0)
    parser.add_argument('--name',               type=str, default="", help="Run name")
    parser.add_argument('--resume',             type=str, default=None, help="Path to experiment directory to resume")
    parser.add_argument('--wandb',              type=str, default=None, help="WandB entity name for logging")
    parser.add_argument('-v', '--verbose',      action="store_true")

    args = parser.parse_args()
    return args

def print_and_log(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'InvFusion:              {c.network_kwargs.joint}')
    dist.print0(f'Dataset:                 {c.dataset_kwargs.class_name}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.fp16}')
    dist.print0()
    dist.print0(f'Print status every:      {c.status // c.batch_size} iters')
    dist.print0(f'Sample images every:     {c.sample_freq // c.status} status updates')
    dist.print0(f'Checkpoint every:        {c.checkpoint_freq // c.status} status updates')

def setup_training_config(args):
    c = dnnlib.EasyDict(**vars(args))

    # Dataset.
    data_class = c.pop("data_class")
    c.dataset_kwargs = dnnlib.EasyDict(class_name=data_class)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels

        del dataset_obj # conserve memory
    except IOError as err:
        raise err

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise Exception(f'--data: Unsupported channel count {dataset_channels}')


    # Hyperparameters.
    degradation = c.pop("degradation")
    blind_train = c.pop("blind_train")
    joint = (degradation is not None) and not blind_train
    c.network_kwargs = dnnlib.EasyDict(class_name="models.hdit.HDiT", joint=joint,
                                       in_mult=2 if blind_train or joint else 1,
                                       data=data_class.split(".")[-1]
                                       )
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.EDM2Loss', P_mean=c.pop("P_mean"),
                                    P_std=c.pop("P_std"), noise_level=c.noise_level,
                                    label_drop=c.pop("label_drop"))
    nppc_loss_weight, nppc_outputs = c.pop("nppc_loss"), c.pop("nppc_outputs")
    if (nppc_loss_weight) > 0:
        c.nppc_loss_weight = nppc_loss_weight
        c.network_kwargs["out_mult"] = 1 + nppc_outputs
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.nppcLoss')
        assert c.mse_train, "--nppc-loss requires --mse-train"
    if c.pop("mse_train"):
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.MSELoss')
        c.sampler_fn = "generate_images.mse"
        if (nppc_loss_weight) > 0:
            c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.nppcLossMSE')

    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=c.pop("lr"),
                                  ref_batches=2 ** c.pop("decay"))
    c.degradation_kwargs = dnnlib.EasyDict(class_name=degradation) if degradation is not None else None
    if (augment := c.pop("augment")):
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    c.wandb = None if c.debug else c.wandb
    c.fp16 = (not c.pop("fp32"))

    for key in ["duration", "status", "checkpoint_freq", "sample_freq"]:
        c[key] = 2 ** c[key]
    return c


def launch_training(run_dir, args):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir) and not args.resume:
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json' if
                args.resume is None else 'training_options_resume.json'), 'wt') as f:
            json.dump(args, f, indent=2)

    wandb_name = args.pop('wandb')
    if dist.get_rank() == 0 and not args.debug and wandb_name:
        try:
            import wandb
            wandb.init(project="invfusion-train", entity=args.wandb, config=args)
            wandb.run.log_code(".")
            args.wandb_log = True
        except ImportError:
            print("wandb not installed, skipping wandb logging.")
    else:
        args.wandb_log = False

    # code save
    if dist.get_rank() == 0:
        code_dir = os.path.join(run_dir, "code" if args.resume is None else "code_resume")
        for path in ["./models", "./training", "./datautils", "./torch_utils", "./dnnlib"] + glob("./*.py"):
            os.makedirs(os.path.dirname(path.replace("./", code_dir + os.path.sep)), exist_ok=True)
            if os.path.isdir(path):
                shutil.copytree(path, path.replace("./", code_dir + os.path.sep))
            else:
                shutil.copy(path, path.replace("./", code_dir + os.path.sep))


    torch.distributed.barrier()
    args.pop("resume")
    args.pop("name")
    training_loop.training_loop(run_dir=run_dir, **args)


def run():
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    args = setup_training_config(args)
    outdir = [os.path.join("experiments", args.name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) if
              args.resume is None else args.resume]
    torch.distributed.broadcast_object_list(outdir, src=0, device=torch.device('cuda'))
    outdir = outdir[0]
    print_and_log(outdir, args)
    launch_training(outdir, args)


if __name__ == "__main__":
    run()