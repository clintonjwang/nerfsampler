import argparse
import inspect
import logging
import os
import pdb
osp = os.path

import torch

import PIL
from PIL import Image
import datasets
from nerfsampler import diffusers
from nerfsampler.diffusers import DDPMScheduler
from nerfsampler.data.core import get_dataset
from tqdm.auto import tqdm
from nerfsampler.networks.ldm import LDM, LatentDiffusionModel
from nerfsampler.networks.prior_transformer import SimpleTransformer as Transformer
from nerfsampler.srt.encoder import ImprovedSRTEncoder
from nerfsampler.srt.decoder import ImprovedSRTDecoder, NerfDecoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
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
        default=18,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
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
        default="tensorboard",
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
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    device = 'cuda'

    # Handle the repository creation
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = denoiser = Transformer(
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        embedding_dim=args.latent_dims,
    ).to(device)

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

        
    train_dataset = get_dataset("train", {'dataset':'msn'})
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    eval_dataset = get_dataset("test", {'dataset':'msn'})
    val_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
    # val_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset, batch_size=1, num_workers=1, 
    #     shuffle=False, pin_memory=False, persistent_workers=True)

    # Potentially load in the weights and states from a previous save
    # Get the most recent checkpoint
    dirs = os.listdir(args.output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    sd = torch.load(os.path.join(args.output_dir, path, 'pytorch_model.bin'))
    model.load_state_dict(sd)

    state_dict = torch.load(osp.expandvars('$NFS/code/nerfsampler/model/isrt_pretrained/model.pt'))
    encoder = ImprovedSRTEncoder(pos_start_octave=-5).to(device)
    encoder.load_state_dict(state_dict['encoder'])
    # state_dict = torch.load(osp.expandvars('$NFS/code/nerfsampler/nerfsampler/srt/runs/msn/nerf/model.pt'))
    # decoder = NerfDecoder(pos_start_octave=-5).to(device)
    decoder = ImprovedSRTDecoder(pos_start_octave=-5).to(device)
    decoder.load_state_dict(state_dict['decoder'])

    model.eval()
    # batch = next(iter(train_dataloader))
    batch = next(iter(val_dataloader))
    input_images = batch.get('input_images').to(device)
    input_camera_pos = batch.get('input_camera_pos').to(device)
    input_rays = batch.get('input_rays').to(device)
    target_pixels = input_images[:,0].reshape(1,-1,3)
    #batch.get('target_pixels').to(device)
    target_camera_pos = input_camera_pos[:,0].tile(1,128*128,1)
    #batch.get('target_camera_pos').to(device)
    target_rays = input_rays[:,0].reshape(1,-1,3)
    #batch.get('target_rays').to(device)
    
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    for i in range(5):
        img = input_images[0,i].cpu().permute(1,2,0).numpy() * 255
        Image.fromarray(img.astype('uint8')).save(osp.join(out_dir, f'input_{i}.png'))

    scheduler=noise_scheduler
    batch_size = 1
    num_inference_steps = 80
    n_latents, latent_dim = 1280, 768
    with torch.no_grad():
        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 5, n_latents, latent_dim)
        latents = torch.randn(latents_shape, device=device)
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            noise_pred = denoiser(latents, t).transformed_states
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = decoder(latents.squeeze(0), target_camera_pos, target_rays)[0]
    
    images -= images.min()
    images /= images.max()
    for i in range(5):
        img = images[i].cpu().reshape(128,128,3).numpy() * 255
        Image.fromarray(img.astype('uint8')).save(osp.join(out_dir, f'rand_{i}.png'))

    with torch.no_grad():
        x = encoder(input_images, input_camera_pos, input_rays)
        clean_latents = x.unsqueeze(0).mean(dim=2, keepdim=True)
        noise = torch.randn(clean_latents.shape).to(clean_latents.device)
        timesteps = torch.full(clean_latents.shape[:1],
            scheduler.timesteps[20], device=clean_latents.device
        ).long()
        latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

        for t in tqdm(scheduler.timesteps[20:]):
            noise_pred = denoiser(latents, t).transformed_states
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = decoder(latents.squeeze(0), target_camera_pos, target_rays)[0]
    
    images -= images.min()
    images /= images.max()
    img = images.squeeze().cpu().reshape(128,128,3).numpy() * 255
    Image.fromarray(img.astype('uint8')).save(osp.join(out_dir, f'recon_{i}.png'))
    return

    # Generate sample images for visual inspection
    pipeline = LatentDiffusionModel(
        denoiser=denoiser,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    pdb.set_trace()
    images = pipeline(
        generator=generator,
        batch_size=args.eval_batch_size,
        output_type="numpy",
    ).images

    # denormalize the images and save to tensorboard
    images_processed = (images * 255).round().astype("uint8")

    if args.logger == "tensorboard":
        accelerator.get_tracker("tensorboard").add_images(
            "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
        )

    clean_latents = encoder(input_images, input_camera_pos, input_rays)
    clean_latents = x.unsqueeze(0).mean(dim=2, keepdim=True) #[B, n_views, num_patches, channels_per_patch]
    # clean_latents = vqvae.encode(x, return_dict=False)

    # Sample noise that we'll add to the images
    noise = torch.randn(clean_latents.shape).to(clean_latents.device)
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (clean_latents.shape[0],), device=clean_latents.device
    ).long()

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

if __name__ == "__main__":
    args = parse_args()
    main(args)
