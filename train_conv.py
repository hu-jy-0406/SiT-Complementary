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
from contextlib import nullcontext
import hashlib
import logging
import os
import random

from models_conv import SiT_models
from transport import create_transport, Sampler
from vae_utils import load_vae
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


def save_checkpoint_atomic(checkpoint, checkpoint_path):
    """Save without exposing a partially-written checkpoint after preemption."""
    temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def capture_local_rng_state(device):
    """Capture every RNG stream used by one training rank."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def gather_rng_states(device):
    """Gather rank-local RNG streams into the rank-zero checkpoint."""
    local_state = capture_local_rng_state(device)
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def restore_local_rng_state(rng_states, device):
    """Restore the RNG streams belonging to this DDP rank."""
    world_size = dist.get_world_size()
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise ValueError(
            "Checkpoint RNG state is incompatible with the current DDP world "
            f"size ({world_size})."
        )
    state = rng_states[dist.get_rank()]
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Checkpoint contains an incomplete per-rank RNG state.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=device)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_lineage(checkpoint, checkpoint_path, resume_step):
    """Persist the true root initialization across later checkpoint hops."""
    if checkpoint is None:
        return {
            "schema_version": 1,
            "mode": "scratch",
            "initial_step": 0,
            "initial_checkpoint": None,
            "initial_checkpoint_sha256": None,
        }
    inherited = checkpoint.get("training_lineage")
    if inherited is not None:
        if not isinstance(inherited, dict):
            raise ValueError("Checkpoint training_lineage must be a mapping.")
        return inherited
    value = None
    if dist.get_rank() == 0:
        resolved = os.path.realpath(checkpoint_path)
        value = {
            "schema_version": 1,
            "mode": "resume",
            "initial_step": int(resume_step),
            "initial_checkpoint": resolved,
            "initial_checkpoint_sha256": _sha256_file(resolved),
        }
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _splitmix64(value):
    """Stable integer mixer used for stateless data-augmentation decisions."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


class RestartSafeImageFolder(ImageFolder):
    """ImageFolder accepting ``(index, flip)`` keys from the sampler."""

    def __getitem__(self, index):
        flip = False
        if isinstance(index, tuple):
            index, flip = index
        sample, target = super().__getitem__(index)
        if flip:
            sample = torch.flip(sample, dims=(-1,))
        return sample, target


class OffsetDistributedSampler(DistributedSampler):
    """Distributed sampler that can resume partway through the first epoch."""

    def __init__(self, *args, deterministic_augmentation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0
        self.deterministic_augmentation = deterministic_augmentation

    def __iter__(self):
        for position, index in enumerate(super().__iter__()):
            if position < self.start_index:
                continue
            if self.deterministic_augmentation:
                global_position = position * self.num_replicas + self.rank
                key = (
                    int(self.seed)
                    ^ ((int(self.epoch) + 1) * 0xD1B54A32D192ED03)
                    ^ ((global_position + 1) * 0x94D049BB133111EB)
                )
                yield index, bool(_splitmix64(key) & 1)
            else:
                yield index

    def __len__(self):
        return max(0, super().__len__() - self.start_index)


def checkpoint_data_state(args, dataset_size, steps_per_epoch, world_size):
    return {
        "dataset_size": dataset_size,
        "steps_per_epoch": steps_per_epoch,
        "world_size": world_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_seed": args.global_seed,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "restart_deterministic_data": args.restart_deterministic_data,
    }


def validate_checkpoint_data_state(saved_state, expected_state):
    if saved_state is None:
        return
    mismatches = {
        key: (saved_state.get(key), value)
        for key, value in expected_state.items()
        if saved_state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint data-resume configuration mismatch: {mismatches}")


def save_training_checkpoint(
    model,
    ema,
    opt,
    args,
    train_steps,
    checkpoint_dir,
    data_state,
    lineage,
    device,
    logger,
    description="checkpoint",
):
    """Collect rank RNG state and atomically commit one resumable checkpoint."""
    rng_states = gather_rng_states(device)
    if dist.get_rank() == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
            "resume_state_version": 1,
            "rng_states": rng_states,
            "data_state": data_state,
            "training_lineage": lineage,
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        save_checkpoint_atomic(checkpoint, checkpoint_path)
        logger.info(f"Saved {description} to {checkpoint_path}")
    dist.barrier()


def infer_checkpoint_step(checkpoint, checkpoint_path, explicit_step=None):
    """Infer an optimizer-step count from old and new training checkpoints."""
    if explicit_step is not None:
        return explicit_step

    if checkpoint.get("train_steps") is not None:
        return int(checkpoint["train_steps"])

    optimizer_steps = {
        int(state["step"])
        for state in checkpoint["opt"].get("state", {}).values()
        if "step" in state
    }
    if len(optimizer_steps) == 1:
        return optimizer_steps.pop()

    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if checkpoint_name.isdigit():
        return int(checkpoint_name)

    raise ValueError(
        "Could not infer the training step from the checkpoint; pass "
        "--resume-step explicitly."
    )


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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    accumulation_divisor = world_size * args.gradient_accumulation_steps
    assert args.global_batch_size % accumulation_divisor == 0, (
        "Global batch size must be divisible by world size times gradient "
        "accumulation steps."
    )
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    micro_batch_size = args.global_batch_size // accumulation_divisor

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = args.run_name or (
            f"{experiment_index:03d}-{model_string_name}-conv-"
            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        )
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = args.wandb_entity or os.environ.get("ENTITY")
            project = args.wandb_project or os.environ.get(
                "PROJECT", "SiT-Complementary"
            )
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    resume_opt_state = None
    resume_train_steps = 0
    resume_rng_states = None
    resume_data_state = None
    loaded_checkpoint = None
    resumed = args.ckpt is not None
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        loaded_checkpoint = checkpoint
        required_keys = {"model", "ema", "opt"}
        if not required_keys.issubset(checkpoint):
            missing = required_keys.difference(checkpoint)
            raise ValueError(f"Training checkpoint is missing keys: {sorted(missing)}")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        resume_opt_state = checkpoint["opt"]
        resume_train_steps = infer_checkpoint_step(
            checkpoint, args.ckpt, args.resume_step
        )
        resume_rng_states = checkpoint.get("rng_states")
        resume_data_state = checkpoint.get("data_state")

    lineage = training_lineage(
        loaded_checkpoint, args.ckpt, resume_train_steps
    )

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
    vae = load_vae(args.vae, device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    if resume_opt_state is not None:
        opt.load_state_dict(resume_opt_state)
        resume_opt_state = None
        loaded_checkpoint = None
        checkpoint = None

    # Setup data:
    transform_steps = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size))
    ]
    if not args.restart_deterministic_data:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )
    transform = transforms.Compose(transform_steps)
    dataset = RestartSafeImageFolder(args.data_path, transform=transform)
    sampler = OffsetDistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
        deterministic_augmentation=args.restart_deterministic_data,
    )
    loader_generator = None
    if args.restart_deterministic_data:
        # Worker base seeds must not consume the training process's CPU RNG.
        # Actual augmentation is stateless, so worker count/prefetch do not
        # affect the samples delivered for a given epoch and sampler position.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed + 0x5EED)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    steps_per_epoch = len(dataset) // args.global_batch_size
    if steps_per_epoch == 0:
        raise ValueError("The dataset is smaller than one global batch.")
    start_epoch = resume_train_steps // steps_per_epoch
    start_step_in_epoch = resume_train_steps % steps_per_epoch
    data_state = checkpoint_data_state(
        args, len(dataset), steps_per_epoch, world_size
    )
    validate_checkpoint_data_state(resume_data_state, data_state)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        "Training configuration: world_size=%d, micro_batch_size=%d, "
        "gradient_accumulation_steps=%d, effective_global_batch_size=%d, "
        "steps_per_epoch=%d",
        world_size,
        micro_batch_size,
        args.gradient_accumulation_steps,
        args.global_batch_size,
        steps_per_epoch,
    )
    logger.info(
        "Resume position: train_step=%d, epoch=%d, step_in_epoch=%d, lr=%g",
        resume_train_steps,
        start_epoch,
        start_step_in_epoch,
        opt.param_groups[0]["lr"],
    )
    logger.info(
        "Restart semantics: deterministic_data=%s, checkpoint_rng=%s",
        args.restart_deterministic_data,
        "present" if resume_rng_states is not None else "legacy-or-new-run",
    )

    # Prepare models for training:
    if not resumed:
        update_ema(ema, model.module, decay=0)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = resume_train_steps
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    sample_batch_size = args.sample_batch_size or micro_batch_size
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
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

    if resumed:
        if resume_rng_states is None:
            logger.warning(
                "Resume checkpoint has no per-rank RNG state. This legacy "
                "continuation cannot reproduce the pre-checkpoint random stream; "
                "new checkpoints will restore the saved logical random streams "
                "under the current world size and deterministic-data setting."
            )
        else:
            restore_local_rng_state(resume_rng_states, device)
            logger.info("Restored per-rank Python/NumPy/CPU/CUDA RNG state.")

    logger.info(f"Training through epoch {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and start_step_in_epoch:
            sampler.start_index = (
                start_step_in_epoch
                * args.gradient_accumulation_steps
                * micro_batch_size
            )
        else:
            sampler.start_index = 0
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            should_update = (
                (micro_step + 1) % args.gradient_accumulation_steps == 0
            )
            sync_context = nullcontext() if should_update else model.no_sync()
            with sync_context:
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += loss.item()

            if not should_update:
                continue

            opt.step()
            opt.zero_grad(set_to_none=True)
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += accumulated_loss / args.gradient_accumulation_steps
            accumulated_loss = 0.0
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = (
                    log_steps * args.global_batch_size / (end_time - start_time)
                )
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"Samples/Sec: {samples_per_sec:.2f}, "
                    f"Peak GPU Memory: {peak_memory_gib:.2f} GiB"
                )
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss,
                            "train steps/sec": steps_per_sec,
                            "train samples/sec": samples_per_sec,
                            "peak gpu memory gib": peak_memory_gib,
                        },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                save_training_checkpoint(
                    model,
                    ema,
                    opt,
                    args,
                    train_steps,
                    checkpoint_dir,
                    data_state,
                    lineage,
                    device,
                    logger,
                )
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample
                    sampled_global_batch_size = sample_batch_size * world_size
                    out_samples = torch.zeros(
                        (
                            sampled_global_batch_size,
                            3,
                            args.image_size,
                            args.image_size,
                        ),
                        device=device,
                    )
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

            if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                break

        # Discard an incomplete accumulation group caused by sampler padding.
        opt.zero_grad(set_to_none=True)

        if args.max_train_steps is not None and train_steps >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if (
        args.save_final_checkpoint
        and train_steps > resume_train_steps
        and train_steps % args.ckpt_every != 0
    ):
        save_training_checkpoint(
            model,
            ema,
            opt,
            args,
            train_steps,
            checkpoint_dir,
            data_state,
            lineage,
            device,
            logger,
            description="final checkpoint",
        )

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
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--restart-deterministic-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Make horizontal flips stateless per epoch/sampler position so a "
            "mid-epoch DDP resume is independent of DataLoader worker prefetch."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
