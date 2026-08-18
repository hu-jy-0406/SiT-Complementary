# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import csv
import importlib
import logging
import math
import os
import re
import shutil
from itertools import islice

MODEL_MODULE_NAME = os.environ.get("SIT_MODEL_MODULE", "models")
model_module = importlib.import_module(MODEL_MODULE_NAME)
SiT_models = model_module.SiT_models
MODEL_IMPLEMENTATION_PATH = os.path.realpath(model_module.__file__)
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class SkipBatchSampler:
    """Skip already-consumed batches without loading or transforming their images."""

    def __init__(self, batch_sampler, skip):
        self.batch_sampler = batch_sampler
        self.skip = skip

    def __iter__(self):
        return islice(iter(self.batch_sampler), self.skip, None)

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.skip)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


@torch.no_grad()
def evaluate_checkpoint_fid(ema, vae, transport_sampler, args, train_steps,
                            device, rank, logger, experiment_dir):
    """Evaluate EMA with CFG=1 while preserving the training RNG trajectory."""
    world_size = dist.get_world_size()
    local_batch = args.fid_per_proc_batch_size
    global_batch = local_batch * world_size
    total_samples = math.ceil(args.fid_num_samples / global_batch) * global_batch
    history_path = args.fid_history or os.path.join(experiment_dir, "fid_cfg1_50k.tsv")
    sample_dir = os.path.join(
        experiment_dir, "fid_cfg1_work", f"{train_steps:07d}"
    )

    # A completed record is reusable after a restart. Rank 0 decides and tells
    # every worker, so all ranks take the same collective path.
    already_done = False
    if rank == 0 and os.path.isfile(history_path):
        with open(history_path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if int(row["step"]) == train_steps and row["status"] == "ok":
                    already_done = True
                    break
    done_tensor = torch.tensor(int(already_done), device=device)
    dist.broadcast(done_tensor, src=0)
    if done_tensor.item():
        logger.info(f"Reusing recorded CFG=1 FID for checkpoint {train_steps:07d}")
        return False

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(args.fid_seed * world_size + rank)
    torch.cuda.manual_seed(args.fid_seed * world_size + rank)

    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        # A prior interrupted attempt may contain a partial sample set.
        for name in os.listdir(sample_dir):
            if name.endswith(".png"):
                os.remove(os.path.join(sample_dir, name))
        logger.info(
            f"Evaluating checkpoint {train_steps:07d}: CFG=1, "
            f"requested={args.fid_num_samples:,}, actual={total_samples:,}"
        )
    dist.barrier()

    sample_fn = transport_sampler.sample_ode(num_steps=args.fid_sampling_steps)
    latent_size = args.image_size // 8
    iterations = total_samples // global_batch
    for batch_index in range(iterations):
        z = torch.randn(local_batch, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (local_batch,), device=device)
        samples = sample_fn(z, ema.forward, y=y)[-1]
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for local_index, sample in enumerate(samples):
            image_index = batch_index * global_batch + local_index * world_size + rank
            Image.fromarray(sample).save(os.path.join(sample_dir, f"{image_index:06d}.png"))
        if batch_index % 10 == 0:
            dist.barrier()
    dist.barrier()

    fid_value = 0.0
    stop_requested = False
    previous_step = None
    previous_fid = None
    if rank == 0:
        from pytorch_fid.fid_score import calculate_fid_given_paths

        fid_value = float(calculate_fid_given_paths(
            [args.fid_reference, sample_dir],
            batch_size=args.fid_inception_batch_size,
            device="cuda:0",
            dims=2048,
            num_workers=args.fid_num_workers,
        ))

        prior_rows = []
        if os.path.isfile(history_path):
            with open(history_path, newline="") as f:
                prior_rows = [
                    row for row in csv.DictReader(f, delimiter="\t")
                    if row["status"] == "ok" and int(row["step"]) < train_steps
                ]
        trend_rows = sorted(
            (
                (int(row["step"]), float(row["fid"]))
                for row in prior_rows
            ),
            key=lambda item: item[0],
        )
        trend_rows.append((train_steps, fid_value))
        required_points = args.fid_stop_consecutive_increases + 1
        recent_trend = trend_rows[-required_points:]
        if prior_rows:
            previous_step, previous_fid = trend_rows[-2]
        if len(recent_trend) == required_points:
            consecutive_increases = all(
                right_fid > left_fid
                for (_, left_fid), (_, right_fid)
                in zip(recent_trend, recent_trend[1:])
            )
            cumulative_rise = recent_trend[-1][1] - recent_trend[0][1]
            required_rise = max(
                args.fid_stop_min_absolute_rise,
                recent_trend[0][1] * args.fid_stop_min_relative_rise,
            )
            stop_requested = consecutive_increases and cumulative_rise >= required_rise

        history_dir = os.path.dirname(os.path.abspath(history_path))
        os.makedirs(history_dir, exist_ok=True)
        needs_header = not os.path.isfile(history_path) or os.path.getsize(history_path) == 0
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            if needs_header:
                writer.writerow([
                    "step", "checkpoint", "status", "fid", "cfg",
                    "num_requested", "num_png", "seed", "timestamp_utc"
                ])
            from datetime import datetime, timezone
            writer.writerow([
                train_steps,
                os.path.join(experiment_dir, "checkpoints", f"{train_steps:07d}.pt"),
                "ok", repr(fid_value), "1.0", args.fid_num_samples,
                total_samples, args.fid_seed,
                datetime.now(timezone.utc).isoformat(),
            ])

        logger.info(f"Checkpoint {train_steps:07d} CFG=1 PyTorch FID: {fid_value:.9f}")
        comparison_output_dir = os.environ.get("SIT_FID_COMPARISON_OUTPUT_DIR")
        if comparison_output_dir:
            try:
                from tools.plot_fid_training_curves import generate_plot
                generated = generate_plot(
                    comparison_output_dir,
                    conv_history=history_path,
                )
                logger.info(
                    f"Updated FID comparison plot: {generated['png']}"
                )
            except Exception:
                # A reporting artifact must never interrupt model training.
                logger.exception("Could not update the FID comparison plot")
        if args.wandb:
            wandb_utils.log({"eval/fid_cfg1_50k": fid_value}, step=train_steps)
        if stop_requested:
            marker = os.path.join(experiment_dir, "FID_REGRESSION_STOPPED")
            with open(marker, "w") as f:
                f.write(
                    f"sustained FID regression over {args.fid_stop_consecutive_increases} "
                    f"consecutive checkpoints: step {recent_trend[0][0]} "
                    f"FID {recent_trend[0][1]:.9f} -> step {train_steps} "
                    f"FID {fid_value:.9f}\n"
                )
            logger.error(
                f"FID increased for {args.fid_stop_consecutive_increases} consecutive "
                f"checkpoints, from {recent_trend[0][1]:.9f} at step "
                f"{recent_trend[0][0]} to {fid_value:.9f}; stopping after "
                f"checkpoint {train_steps:07d}."
            )
        shutil.rmtree(sample_dir)

    result = torch.tensor([fid_value, float(stop_requested)], device=device)
    dist.broadcast(result, src=0)
    dist.barrier()

    # Evaluation must not perturb the random stream used by resumed training.
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    return bool(result[1].item())


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    expected_model_module = os.environ.get("SIT_EXPECTED_MODEL_MODULE")
    if expected_model_module and MODEL_MODULE_NAME != expected_model_module:
        raise RuntimeError(
            f"Expected model module {expected_model_module!r}, but loaded "
            f"{MODEL_MODULE_NAME!r} from {MODEL_IMPLEMENTATION_PATH}"
        )

    # Load resume metadata before creating the output directory or WandB run. The
    # checkpoint hyperparameters remain authoritative; only runtime location,
    # target epoch, and logging options may be overridden by the command line.
    resume_checkpoint = None
    resume_step = 0
    if args.ckpt is not None:
        runtime_args = args
        resume_checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        checkpoint_args = resume_checkpoint["args"]
        runtime_names = (
            "data_path", "results_dir", "epochs", "wandb", "ckpt", "run_name",
            "fid_every_checkpoint", "fid_every", "fid_num_samples", "fid_reference",
            "fid_history", "fid_per_proc_batch_size", "fid_inception_batch_size",
            "fid_num_workers", "fid_sampling_steps", "fid_seed",
            "fid_stop_consecutive_increases", "fid_stop_min_absolute_rise",
            "fid_stop_min_relative_rise",
        )
        if not hasattr(checkpoint_args, "learning_rate"):
            checkpoint_args.learning_rate = runtime_args.learning_rate
        for name in runtime_names:
            setattr(checkpoint_args, name, getattr(runtime_args, name))
        args = checkpoint_args
        match = re.fullmatch(r"(\d+)\.pt", os.path.basename(args.ckpt))
        if match is None:
            raise ValueError("Cannot infer the training step from checkpoint filename; expected NNNNNNN.pt")
        resume_step = int(match.group(1))

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = args.run_name or (f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}")
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(
            f"Model implementation: {MODEL_MODULE_NAME} ({MODEL_IMPLEMENTATION_PATH})"
        )

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)
        experiment_dir = None
    path_objects = [experiment_dir]
    dist.broadcast_object_list(path_objects, src=0)
    experiment_dir = path_objects[0]

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    logger.info(
        f"Optimizer: AdamW(lr={args.learning_rate:g}, weight_decay=0, "
        "betas=(0.9, 0.999))"
    )
    if resume_checkpoint is not None:
        model.module.load_state_dict(resume_checkpoint["model"])
        ema.load_state_dict(resume_checkpoint["ema"])
        opt.load_state_dict(resume_checkpoint["opt"])
        logger.info(f"Resumed model, EMA, and optimizer from {args.ckpt} at step {resume_step:,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    if resume_checkpoint is None:
        update_ema(ema, model.module, decay=0)  # Initialize EMA only for a new run.
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    steps_per_epoch = len(loader)
    target_steps = args.epochs * steps_per_epoch
    start_epoch, batches_to_skip = divmod(train_steps, steps_per_epoch)
    logger.info(
        f"Training to {args.epochs} total epochs ({target_steps:,} steps); "
        f"starting at epoch {start_epoch}, batch {batches_to_skip}, step {train_steps:,}."
    )
    # Establish a same-protocol 50k baseline for the resume checkpoint before
    # comparing later checkpoints against it. A recorded baseline is reused.
    if args.fid_every_checkpoint and resume_checkpoint is not None:
        if evaluate_checkpoint_fid(
            ema, vae, transport_sampler, args, train_steps,
            device, rank, logger, experiment_dir,
        ):
            logger.error("Resume checkpoint already violates the recorded FID trend; exiting.")
            cleanup()
            return
        start_time = time()

    stop_requested = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        epoch_loader = loader
        if epoch == start_epoch and batches_to_skip:
            epoch_loader = DataLoader(
                dataset,
                batch_sampler=SkipBatchSampler(loader.batch_sampler, batches_to_skip),
                num_workers=args.num_workers,
                pin_memory=True,
            )
        for x, y in epoch_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
                if args.fid_every_checkpoint and train_steps % args.fid_every == 0:
                    stop_requested = evaluate_checkpoint_fid(
                        ema, vae, transport_sampler, args, train_steps,
                        device, rank, logger, experiment_dir,
                    )
                    start_time = time()
                    if stop_requested:
                        break
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

        if stop_requested:
            break

    if rank == 0 and not stop_requested and train_steps % args.ckpt_every != 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved final checkpoint to {checkpoint_path}")
    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Experiment directory and WandB run name (useful when resuming)")
    parser.add_argument("--fid-every-checkpoint", action="store_true",
                        help="Run periodic CFG=1 PyTorch FID checks and stop on sustained regression")
    parser.add_argument("--fid-every", type=int, default=50_000,
                        help="Training-step interval between FID checks")
    parser.add_argument("--fid-num-samples", type=int, default=50_000)
    parser.add_argument("--fid-reference", type=str, default=None,
                        help="Reference .npz required when --fid-every-checkpoint is enabled")
    parser.add_argument("--fid-history", type=str, default=None)
    parser.add_argument("--fid-per-proc-batch-size", type=int, default=64)
    parser.add_argument("--fid-inception-batch-size", type=int, default=128)
    parser.add_argument("--fid-num-workers", type=int, default=8)
    parser.add_argument("--fid-sampling-steps", type=int, default=250)
    parser.add_argument("--fid-seed", type=int, default=0)
    parser.add_argument("--fid-stop-consecutive-increases", type=int, default=3,
                        help="Stop only after this many consecutive checkpoint FID increases")
    parser.add_argument("--fid-stop-min-absolute-rise", type=float, default=0.25,
                        help="Minimum cumulative absolute FID rise required to stop")
    parser.add_argument("--fid-stop-min-relative-rise", type=float, default=0.005,
                        help="Minimum cumulative relative FID rise required to stop")

    parse_transport_args(parser)
    args = parser.parse_args()
    if args.fid_every_checkpoint and not args.fid_reference:
        parser.error("--fid-reference is required with --fid-every-checkpoint")
    main(args)
