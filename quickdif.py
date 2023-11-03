COLS_XL = {
    'black': [-2.8232955932617188, 0.5033664703369141, 0.3139435052871704, 0.3359411954879761],
    'white': [2.3501884937286377, 0.224839448928833, 1.2127048969268799, -1.0597002506256104],
    'red': [-2.561439037322998, -2.5784878730773926, 1.3915523290634155, -1.6186809539794922],
    'green': [-0.45990604162216187, 1.8333771228790283, 3.4502267837524414, 1.1301288604736328],
    'blue': [0.05935399606823921, 2.129012107849121, -2.3017752170562744, 0.5399889349937439],
    'cyan': [1.619562029838562, 3.388197183609009, 0.5599392056465149, 1.0360429286956787],
    'magenta': [-0.12525132298469543, -0.6654946208000183, -1.5711115598678589, -1.175052523612976],
    'yellow': [-0.8608582019805908, -1.375950574874878, 4.230403900146484, -1.0693323612213135]
}
COLS_FTMSE = {
    'black': [-0.9953256249427795, -2.602407455444336, 1.1153717041015625, 1.2966053485870361],
    'white': [2.1749563217163086, 1.4434212446212769, -0.03183012455701828, -1.162193775177002],
    'red': [1.2575085163116455, -0.8768408298492432, -1.8788076639175415, 0.7792190313339233],
    'green': [0.768126904964447, 0.821873128414154, 2.3520827293395996, 1.8135154247283936],
    'blue': [-0.6330060958862305, -3.0466079711914062, 0.9888341426849365, -1.4022014141082764],
    'cyan': [0.5828086137771606, -0.10415874421596527, 2.934731960296631, -0.09788236021995544],
    'magenta': [0.7387062311172485, -1.555786371231079, -1.3593989610671997, -1.438640832901001],
    'yellow': [2.3476614952087402, 2.0031352043151855, -0.079119011759758, 1.6609115600585938]
}

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
parser.add_argument('-b', '--batch-count', type=int, default=1)
parser.add_argument('-B', '--batch-size', type=int, default=1)
parser.add_argument('-c', '--color', choices=list(COLS_XL.keys()), default='black')
parser.add_argument('-C', '--color-scale', type=float, default=0.0)
parser.add_argument('-m', '--model', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument('-o', '--out', type=Path, default="/tmp/quickdif/")
parser.add_argument('-d', '--dtype', choices=["fp16", "bf16", "fp32"], default="fp16")
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('-S', '--sampler', choices=['default', 'dpm', 'ddim', 'ddimp', 'euler'], default='default')
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
    EulerDiscreteScheduler,
)
from compel import Compel, ReturnedEmbeddingsType

torch.set_float32_matmul_precision('high')
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

pipe_args = {'torch_dtype': dtype, 'use_safetensors': True, 'add_watermarker': False}

if args.model.endswith('.safetensors'):
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(args.model, **pipe_args)
        XL = True
    except:
        pipe = StableDiffusionPipeline.from_single_file(args.model, **pipe_args)
        XL = False
else:
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model, **pipe_args)
        XL = True
    except:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, **pipe_args)
        XL = False

if XL:
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False)
else:
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False)

pipe.safety_checker = None
if dtype != torch.float16:
    pipe.vae.force_upcast = False

if args.compile:
    pipe.unet = torch.compile(pipe.unet)
else:
    pipe.unet.set_default_attn_processor()
    pipe.vae.set_default_attn_processor()

sigma = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing').init_noise_sigma

if args.sampler == "dpm":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type='dpmsolver++',
        use_karras_sigmas=True,
    )
elif args.sampler == "ddim":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif args.sampler == "ddimp":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print("PATCHING TIMESTEPS")
    tsbase = pipe.scheduler.set_timesteps

    def tspatch(*args, **kwargs):
        if 'num_inference_steps' in kwargs:
            num_inference_steps = kwargs['num_inference_steps']
            num_inference_steps -= 1
            kwargs['num_inference_steps'] = num_inference_steps
        else:
            args=list(args)
            num_inference_steps = args[0]
            num_inference_steps -= 1
            args[0] = num_inference_steps
        tsbase(*args, **kwargs)
        pipe.scheduler.timesteps = torch.cat([pipe.scheduler.timesteps, pipe.scheduler.timesteps[-1].remainder(num_inference_steps).reshape(1)])

    pipe.scheduler.set_timesteps = tspatch
elif args.sampler == "euler":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.scheduler.config.timestep_spacing='trailing'


size = [1, 4, args.height // 8, args.width // 8]

if args.color_scale <= 0.0:
    latent_input = torch.zeros(size, dtype=dtype, device='cpu')
else:
    latent_input = torch.tensor(COLS_XL[args.color] if XL else COLS_FTMSE[args.color], dtype=dtype,
                           device='cpu').mul(args.color_scale).div(sigma).expand([size[0], size[2], size[3], size[1]]).permute((0, 3, 1, 2)).clone()

size[0] = args.batch_size

pipe.to('cuda')

n = 0
for (pn, prompt) in enumerate(args.prompts * args.batch_count):
    # add noise
    latents = latent_input.expand(size).clone()
    for (ln, latent) in enumerate(latents):
        seed = (args.seed if args.seed >= 0 else torch.randint(2**31-1).item()) + pn * args.batch_size + ln
        print(f"seed:{seed}")
        generator = torch.manual_seed(seed)
        # f32 noise for equal seeds amongst other UIs
        latent += torch.randn(latents.shape[1:], generator=generator, dtype=torch.float32)
    kwargs = {
        "width": args.width,
        "height": args.height,
        "latents": latents,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
        "guidance_rescale": args.rescale,
        "num_images_per_prompt": args.batch_size,
    }

    # compel
    if XL:
        ncond, npool = compel.build_conditioning_tensor(args.negative)
        pcond, ppool = compel.build_conditioning_tensor(prompt)
        kwargs = kwargs | {'pooled_prompt_embeds': ppool, 'negative_pooled_prompt_embeds': npool}
    else:
        pcond = compel.build_conditioning_tensor(prompt)
        ncond = compel.build_conditioning_tensor(args.negative)

    pcond, ncond = compel.pad_conditioning_tensors_to_same_length([pcond, ncond])
    kwargs = kwargs | {'prompt_embeds': pcond, 'negative_prompt_embeds': ncond}

    # compute + save
    for image in pipe(**kwargs).images:
        p = args.out.joinpath(f"{n:05}.png")
        while p.exists():
            n += 1
            p = args.out.joinpath(f"{n:05}.png")

        image.save(p, format="PNG")
        del image, p

    # fix memleak
    del kwargs, latents, pcond, ncond
    if XL:
        del ppool, npool
    if (lambda f, t: f / t)(*torch.cuda.mem_get_info()) < 0.25:
        gc.collect()
        torch.cuda.empty_cache()
