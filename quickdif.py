COLS_XL = {'black': [-0.1795562356710434, 0.032013148069381714, 0.019966214895248413, 0.021365217864513397], 'white': [0.14946752786636353, 0.014299359172582626, 0.07712572813034058, -0.0673949271440506], 'red': [-0.16290265321731567, -0.163986936211586, 0.08850008249282837, -0.10294502973556519], 'green': [-0.029249146580696106, 0.11659930646419525, 0.21942783892154694, 0.07187405228614807], 'blue': [0.0037747935857623816, 0.13540109992027283, -0.14638851583003998, 0.034342244267463684], 'cyan': [0.10300105810165405, 0.2154828906059265, 0.03561106696724892, 0.0658903494477272], 'magenta': [-0.007965748198330402, -0.04232420772314072, -0.09991972893476486, -0.07473109662532806], 'yellow': [-0.05474894493818283, -0.08750784397125244, 0.2690456509590149, -0.06800749897956848]}
COLS_FTMSE = {'black': [-0.06621697545051575, -0.17313283681869507, 0.07420338690280914, 0.08626051247119904], 'white': [0.14469541609287262, 0.09602789580821991, -0.0021176044829189777, -0.07731839269399643], 'red': [0.08365947008132935, -0.05833444371819496, -0.12499324232339859, 0.05183985084295273], 'green': [0.051101915538311005, 0.0546775683760643, 0.15647926926612854, 0.12064949423074722], 'blue': [-0.04211260378360748, -0.20268462598323822, 0.06578511744737625, -0.09328559786081314], 'cyan': [0.038773078471422195, -0.00692947581410408, 0.19524171948432922, -0.006511914078146219], 'magenta': [0.04914461076259613, -0.10350330919027328, -0.09043803811073303, -0.09570984542369843], 'yellow': [0.156185120344162, 0.13326449692249298, -0.005263628903776407, 0.11049704253673553]}


import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Custom diffusers models', add_help=False)
parser.add_argument('prompts', nargs='+', type=str)
parser.add_argument('-n', '--negative', type=str, default="blurry, cropped, text")
parser.add_argument('-w', '--width', type=int, default=1024)
parser.add_argument('-h', '--height', type=int, default=1024)
parser.add_argument('-s', '--steps', type=int, default=30)
parser.add_argument('-g', '--cfg', type=float, default=8.0)
parser.add_argument('-G', '--rescale', type=float, default=0.7)
parser.add_argument('-c', '--color', choices=list(COLS_XL.keys()), default='black')
parser.add_argument('-C', '--color_scale', type=float, default=0.0)
parser.add_argument('-m', '--model', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument('-o', '--out', type=Path, default="/tmp/quickdif/")
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--dpm', action='store_true')
parser.add_argument('--compile', action='store_true')
parser.add_argument('--help', action='help')

args = parser.parse_args()

if args.out.is_dir():
    pass
elif not args.out.exists():
    args.out.mkdir()
else:
    raise ValueError("out must be directory")

import torch, gc
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
torch.set_float32_matmul_precision('high')

pipe_args = {'torch_dtype':torch.float16, 'use_safetensors':True, 'add_watermarker':False}

if args.model.endswith('.safetensors'):
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(args.model, **pipe_args)
        XL=True
    except:
        pipe = StableDiffusionPipeline.from_single_file(args.model, **pipe_args)
        XL=False
else:
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model, **pipe_args)
        XL=True
    except:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, **pipe_args)
        XL=False

if XL:
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True], truncate_long_prompts= False)
else:
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts= False)

pipe.safety_checker = None

if args.compile:
    pipe.unet = torch.compile(pipe.unet)
else:
    pipe.unet.set_default_attn_processor()
    pipe.vae.set_default_attn_processor()

if args.dpm:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing='trailing',
        algorithm_type='dpmsolver++',
        use_karras_sigmas=True,
    )
else:
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing='trailing',
    )

generator = torch.manual_seed(args.seed) if args.seed >= 0 else torch.default_generator

size = [1, 4, args.height // 8, args.width // 8]

if args.color_scale <= 0.0:
    latents = torch.zeros(size, dtype=torch.float16, device='cpu')
else:
    latents = torch.tensor(COLS_XL[args.color] if XL else COLS_FTMSE[args.color], dtype=torch.float16, device='cpu').mul(args.color_scale).expand([size[0], size[2], size[3], size[1]]).permute((0, 3, 1, 2)).clone()

# f32 noise for equal seeds amongst other UIs
latents += torch.randn(latents.shape, generator=generator, dtype=torch.float32)


pipe.to('cuda')

n=0
for prompt in args.prompts:
    kwargs = {
    "width":args.width,
    "height":args.height,
    "latents":latents,
    "num_inference_steps":args.steps,
    "guidance_scale":args.cfg,
    "guidance_rescale":args.rescale,
    }

    if XL:
        ncond, npool = compel.build_conditioning_tensor(args.negative)
        pcond, ppool = compel.build_conditioning_tensor(prompt)
        kwargs = kwargs | {'pooled_prompt_embeds':ppool, 'negative_pooled_prompt_embeds':npool}
    else:
        pcond = compel.build_conditioning_tensor(prompt)
        ncond = compel.build_conditioning_tensor(args.negative)

    pcond, ncond = compel.pad_conditioning_tensors_to_same_length([pcond, ncond])
    kwargs = kwargs | {'prompt_embeds':pcond, 'negative_prompt_embeds':ncond}
    image = pipe(**kwargs).images[0]

    p = args.out.joinpath(f"{n:05}.png")
    while p.exists():
        n += 1
        p = args.out.joinpath(f"{n:05}.png")

    image.save(p, format="PNG")

    del image, p, kwargs, pcond, ncond
    if XL: del ppool, npool
    if (lambda f,t: f/t)(*torch.cuda.mem_get_info()) < 0.25:
        gc.collect()
        torch.cuda.empty_cache()
