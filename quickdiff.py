import argparse

parser = argparse.ArgumentParser(description='Custom diffusers models', add_help=False)
parser.add_argument('prompt', type=str)
parser.add_argument('-n', '--negative', type=str, default="blurry, cropped, text")
parser.add_argument('-w', '--width', type=int, default=1024)
parser.add_argument('-h', '--height', type=int, default=1024)
parser.add_argument('-s', '--steps', type=int, default=30)
parser.add_argument('-g', '--cfg', type=float, default=6.0)
parser.add_argument('-G', '--rescale', type=float, default=0.7)
parser.add_argument('-m', '--model', type=str, default="ptx0/vanilla-xltest")
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--help', action='help')

args = parser.parse_args()

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
)

pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, use_safetensors=True)

if args.compile:
    pipe.unet = torch.compile(pipe.unet)

scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    rescale_betas_zero_snr=True,
)

generator = torch.manual_seed(args.seed) if args.seed >= 0 else torch.default_generator

image = pipe.to('cuda')(
    prompt=args.prompt,
    width=args.width,
    height=args.height,
    negative_prompt=args.negative,
    generator=generator,
    num_images_per_prompt=1,
    num_inference_steps=args.steps,
    guidance_scale=args.cfg,
    guidance_rescale=args.rescale,
).images[0]

image.save("/tmp/test.png", format="PNG")
