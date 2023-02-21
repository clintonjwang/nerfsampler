import argparse
import inspect
import logging
import math
import os
import pdb
osp = os.path

import torch
import torch.nn.functional as F

import datasets
from nerfsampler import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from nerfsampler.diffusers import DDPMScheduler
from nerfsampler.diffusers.models.vq_model import VQModel
from nerfsampler.diffusers.optimization import get_scheduler
from nerfsampler.data.core import get_dataset
from nerfsampler.diffusers.training_utils import EMAModel
# from torchvision.transforms import (
#     CenterCrop,
#     Compose,
#     InterpolationMode,
#     Normalize,
#     RandomHorizontalFlip,
#     Resize,
#     ToTensor,
# )
from tqdm.auto import tqdm
from nerfsampler.networks.ldm import LDM, LatentDiffusionModel
from nerfsampler.networks.prior_transformer import SimpleTransformer as Transformer
from nerfsampler.srt.encoder import ImprovedSRTEncoder
from nerfsampler.srt.decoder import NerfDecoder


logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--eval_epochs", type=int, default=1, help="How often to eval.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    # vqvae = VQModel(latent_channels=args.latent_dims)
    model = denoiser = Transformer(
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        embedding_dim=args.latent_dims,
    ) #in_channels=args.latent_dims, out_channels=args.latent_dims, 
    # model = LDM(vqvae, denoiser)
    model.enable_xformers_memory_efficient_attention()
    pdb.set_trace()

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
        )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    device = accelerator.device
        
    # Preprocessing the datasets and DataLoaders creation.
    # augmentations = Compose(
    #     [
    #         Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
    #         CenterCrop(args.resolution),
    #         RandomHorizontalFlip(),
    #         ToTensor(),
    #         Normalize([0.5], [0.5]),
    #     ]
    # )
    # def transforms(examples):
    #     images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    #     return {"input": images}

    batch_size = args.train_batch_size
    train_dataset = get_dataset("train", {'dataset':'msn'})
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.dataloader_num_workers, pin_memory=True,
        shuffle=False, persistent_workers=args.dataloader_num_workers > 0)
    eval_dataset = get_dataset("val", {'dataset':'msn'}, max_len=128)
    val_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, num_workers=1, 
        shuffle=False, pin_memory=False, persistent_workers=True)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        accelerator.register_for_checkpointing(ema_model)
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    state_dict = torch.load(osp.expandvars('$NFS/code/nerfsampler/model/isrt_pretrained/model.pt'))
    encoder = ImprovedSRTEncoder(pos_start_octave=-5).to(device)
    encoder.load_state_dict(state_dict['encoder'])
    # decoder = NerfDecoder().to(device)
    # decoder.load_state_dict(state_dict['decoder'])
    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            break
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            input_images = batch.get('input_images').to(device)
            input_camera_pos = batch.get('input_camera_pos').to(device)
            input_rays = batch.get('input_rays').to(device)
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                x = encoder(input_images, input_camera_pos, input_rays)
            clean_latents = x.unsqueeze(0).mean(dim=2, keepdim=True) #[B, n_views, num_patches, channels_per_patch]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_latents.shape).to(clean_latents.device)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_latents.shape[0],), device=clean_latents.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                outputs = model(
                    hidden_states=noisy_latents,
                    timestep=timesteps).transformed_states

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(outputs, noise)  # this could have different weights!
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_latents.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        outputs, clean_latents, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.eval_epochs == 0 or epoch == args.num_epochs - 1:
                model.eval()
                for batch in val_dataloader:
                    input_images = batch.get('input_images').to(device)
                    input_camera_pos = batch.get('input_camera_pos').to(device)
                    input_rays = batch.get('input_rays').to(device)
                    
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        x = encoder(input_images, input_camera_pos, input_rays)
                        clean_latents = x.unsqueeze(0).mean(dim=2, keepdim=True) #[B, n_views, num_patches, channels_per_patch]
                        noise = torch.randn(clean_latents.shape).to(clean_latents.device)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (clean_latents.shape[0],), device=clean_latents.device
                        ).long()
                        noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
                        outputs = model(
                            hidden_states=noisy_latents,
                            timestep=timesteps).transformed_states
                        if args.prediction_type == "epsilon":
                            loss = F.mse_loss(outputs, noise)
                        elif args.prediction_type == "sample":
                            alpha_t = _extract_into_tensor(
                                noise_scheduler.alphas_cumprod, timesteps, (clean_latents.shape[0], 1, 1, 1)
                            )
                            snr_weights = alpha_t / (1 - alpha_t)
                            loss = snr_weights * F.mse_loss(
                                outputs, clean_latents, reduction="none"
                            )  # use SNR weighting from distillation paper
                            loss = loss.mean()
                        logs = {"val loss": loss.item()}
                        accelerator.log(logs, step=global_step)
                model.train()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
