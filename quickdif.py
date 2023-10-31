COLS_XL = {'black': [-0.17645598948001862, 0.03146037831902504, 0.019621476531028748, 0.02099631540477276], 'white': [0.14688679575920105, 0.014052464626729488, 0.07579405605792999, -0.06623126566410065], 'red': [-0.16008996963500977, -0.16115550696849823, 0.08697202056646347, -0.10116755962371826], 'green': [-0.028744127601385117, 0.11458607017993927, 0.2156391739845276, 0.07063306123018265], 'blue': [0.003709609853103757, 0.13306322693824768, -0.14386092126369476, 0.0337492972612381], 'cyan': [0.10122263431549072, 0.21176232397556305, 0.03499620035290718, 0.06475268304347992], 'magenta': [-0.007828177884221077, -0.041593436151742935, -0.09819445759057999, -0.07344077527523041], 'yellow': [-0.053803637623786926, -0.08599691092967987, 0.2644002437591553, -0.06683327257633209]}
COLS_FTMSE = {'black': [-0.06220785155892372, -0.1626504510641098, 0.06971073150634766, 0.08103783428668976], 'white': [0.1359347701072693, 0.0902138352394104, -0.0019893920980393887, -0.07263711094856262], 'red': [0.07859428226947784, -0.054802559316158295, -0.11742548644542694, 0.04870118200778961], 'green': [0.04800793528556824, 0.051367077976465225, 0.14700517058372498, 0.1133447140455246], 'blue': [-0.039562880992889404, -0.1904129981994629, 0.06180213391780853, -0.08763758838176727], 'cyan': [0.03642553836107254, -0.006509925238788128, 0.18342071771621704, -0.006117651239037514], 'magenta': [0.046169131994247437, -0.09723665565252304, -0.08496242761611938, -0.08991505205631256], 'yellow': [0.14672884345054626, 0.1251959651708603, -0.004944938700646162, 0.10380695760250092]}


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
