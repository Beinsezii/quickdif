XL_CONSTS = {
    "black" : [-21.675981521606445, 3.864609956741333, 2.4103028774261475, 2.579195261001587],
    "white" : [18.043685913085938, 1.7262177467346191, 9.310612678527832, -8.135881423950195],
    "red" : [-19.665550231933594, -19.79644012451172, 10.68371868133545, -12.427474021911621],
    "green" : [-3.530947685241699, 14.075841903686523, 26.489261627197266, 8.67661190032959],
    "blue" : [0.45569008588790894, 16.3455867767334, -17.67197036743164, 4.145791053771973],
    "cyan" : [12.434264183044434, 26.013031005859375, 4.298962593078613, 7.954266548156738],
    "magenta" : [-0.9616246223449707, -5.109368801116943, -12.062283515930176, -9.02152156829834],
    "yellow" : [-6.609264373779297, -10.563915252685547, 32.47910690307617, -8.209832191467285],
}

import argparse

parser = argparse.ArgumentParser(description='Custom diffusers models', add_help=False)
parser.add_argument('prompt', type=str)
parser.add_argument('-n', '--negative', type=str, default="blurry, cropped, text")
parser.add_argument('-w', '--width', type=int, default=1024)
parser.add_argument('-h', '--height', type=int, default=1024)
parser.add_argument('-s', '--steps', type=int, default=30)
parser.add_argument('-g', '--cfg', type=float, default=6.0)
parser.add_argument('-G', '--rescale', type=float, default=0.7)
parser.add_argument('-c', '--color', choices=list(XL_CONSTS.keys()), default='black')
parser.add_argument('-C', '--color_scale', type=float, default=0.0)
parser.add_argument('-m', '--model', type=str, default="ptx0/coco-xltest")
parser.add_argument('-o', '--out', type=argparse.FileType('wb'), default="/tmp/quickdiff.png")
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--help', action='help')

args = parser.parse_args()

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
)

generator = torch.manual_seed(args.seed) if args.seed >= 0 else torch.default_generator

size = [1, 4, args.height // 8, args.width // 8]

if args.color_scale <= 0.0:
    latents = torch.zeros(size, dtype=torch.float16, device='cpu')
else:
    latents = torch.tensor(XL_CONSTS[args.color], dtype=torch.float16, device='cpu').div(112.5).mul(args.color_scale).expand([size[0], size[2], size[3], size[1]]).permute((0, 3, 1, 2)).clone()

latents += torch.randn(latents.shape, generator=generator)

if args.model.endswith('.safetensors'):
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False
    )
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False
    )

if args.compile:
    pipe.unet = torch.compile(pipe.unet)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

image = pipe.to('cuda')(
    prompt=args.prompt,
    width=args.width,
    height=args.height,
    negative_prompt=args.negative,
    latents=latents,
    num_inference_steps=args.steps,
    guidance_scale=args.cfg,
    guidance_rescale=args.rescale,
).images[0]

image.save(args.out, format="PNG")
