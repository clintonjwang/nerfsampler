import shutil
import torch, sys, os, argparse
from nerfsampler.diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    #python scripts/inpaint.py -i=0
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-s', '--speed', type=int, default=4)
    parser.add_argument('--pan_direction', default='right')
    parser.add_argument('--overlap_thresh', default=20)
    parser.add_argument('--separation', default=300)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")

    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to('cuda')

    prompt = ["an elegant bedroom in a giant tree, with branches and vines covering the walls and ceiling",]

    n_views = 64
    for view_ix in range(n_views):
        shutil.rmtree(f"./out{p_ix}", ignore_errors=True)
        keys = ordering[p_ix], ordering[p_ix+1]
        root = './keyframes/'

        p_ix = int(p_ix)
        folder = f'out{p_ix}'

        cur_image = pil_to_tensor(Image.open(root+keys[0]+'.png').convert('RGB').resize((512,512), resample=Image.Resampling.LANCZOS))
        keyframe2 = pil_to_tensor(Image.open(root+keys[1]+'.png').convert('RGB').resize((512,512), resample=Image.Resampling.LANCZOS))

        neg_prompt = 'blurry, multiple rooms, artifacts, border'

        joined_image = torch.cat((cur_image,
            cur_image.new_zeros(3, cur_image.size(1), args.separation),
            keyframe2), dim=-1)
        total_width = joined_image.size(-1)

        dx = (total_width - 512)//2
        crop = joined_image[...,dx:-dx]
        mask = torch.zeros_like(crop)
        mask[...,512-dx:-512+dx] = 255
        prompt = "to the left, " + prompts[keys[0]] + ", smoothly transitioning, to the right, " + prompts[keys[1]] + ", beautiful vivid high resolution"
        with torch.autocast('cuda'):
            output = pipe(prompt=prompt,
                image=to_pil_image(crop), mask_image=to_pil_image(mask),
                num_inference_steps=200, 
                negative_prompt=neg_prompt, guidance_scale=7,
            ).images[0]
        joined_image[...,dx:-dx] = pil_to_tensor(output)
