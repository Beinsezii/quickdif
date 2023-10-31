# COLS_XL = {
#     "black" : [-21.675981521606445, 3.864609956741333, 2.4103028774261475, 2.579195261001587],
#     "white" : [18.043685913085938, 1.7262177467346191, 9.310612678527832, -8.135881423950195],
#     "red" : [-19.665550231933594, -19.79644012451172, 10.68371868133545, -12.427474021911621],
#     "green" : [-3.530947685241699, 14.075841903686523, 26.489261627197266, 8.67661190032959],
#     "blue" : [0.45569008588790894, 16.3455867767334, -17.67197036743164, 4.145791053771973],
#     "cyan" : [12.434264183044434, 26.013031005859375, 4.298962593078613, 7.954266548156738],
#     "magenta" : [-0.9616246223449707, -5.109368801116943, -12.062283515930176, -9.02152156829834],
#     "yellow" : [-6.609264373779297, -10.563915252685547, 32.47910690307617, -8.209832191467285],
# }

COLS_XL = {'black': [-5.54072403911232, 4.526814351169378, 3.550338567178642, -4.3354098796480685], 'white': [18.00428923824802, 1.7538797715308192, 9.681977973719768, -7.940033342878451], 'red': [-7.282218298323642, -11.132304124286748, 0.5311491123161431, -14.735129391861847], 'green': [1.9356177158456376, 10.37097296048887, 18.36263093616435, 0.7188784081624817], 'blue': [3.8997846263082465, 9.811691337861703, -8.77209970181866, -2.86256140756484], 'cyan': [12.624110033444595, 17.35059885602095, 6.341289376941859, 1.3279199944506672], 'magenta': [2.867751094221603, -4.095526434520252, -6.159327863522776, -10.434268915982102], 'yellow': [1.8804551483053729, -5.611284743616125, 23.825928250560537, -8.86957961715234]}

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

generator = torch.manual_seed(args.seed) if args.seed >= 0 else torch.default_generator

size = [1, 4, args.height // 8, args.width // 8]

if args.color_scale <= 0.0:
    latents = torch.zeros(size, dtype=torch.float16, device='cpu')
else:
    latents = torch.tensor(COLS_XL[args.color], dtype=torch.float16, device='cpu').mul(args.color_scale).expand([size[0], size[2], size[3], size[1]]).permute((0, 3, 1, 2)).clone()

# f32 noise for equal seeds amongst other UIs
latents += torch.randn(latents.shape, generator=generator, dtype=torch.float32)

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
