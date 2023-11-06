# LATENT COLORS {{{
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
# LATENT COLORS }}}

# CLI {{{
import argparse, os
from pathlib import Path

outdefault = '/tmp/' if os.path.exists('/tmp/') else './output/'
mdefault = "stabilityai/stable-diffusion-xl-base-1.0"
samplers = ['dpm', "ddim", "ddimp", "euler"]
dtypes = ["fp16", "bf16", "fp32"]
offload = ["none", "model", "sequential"]

parser = argparse.ArgumentParser(description="Quick and easy inference for a variety of Diffusers models. Not all models support all options",
                                 add_help=False)
parser.add_argument('prompts', nargs='+', type=str)
parser.add_argument('-n', '--negative', type=str, nargs='+', default=["blurry, cropped, text"], help="Universal negative for all prompts")
parser.add_argument('-w', '--width', type=int, help="Final output width. Default varies by model")
parser.add_argument('-h', '--height', type=int, help="Final output height. Default varies by model")
parser.add_argument('-s', '--steps', type=int, nargs='*', default=[30], help="Number of inference steps. Default 30. Can be unset")
parser.add_argument('-g', '--cfg', type=float, nargs='*', default=[8], help="Guidance for conditioning. Default 8. Can be unset")
parser.add_argument('-G', '--rescale', type=float, nargs='*', default=[0.7], help="Guidance rescale factor. Default 0.7. Can be unset")
parser.add_argument('-b', '--batch-count', type=int, default=1, help="Amount of times to run each prompt sequentially. Default 1")
parser.add_argument('-B', '--batch-size', type=int, default=1, help="Amount of times to run each prompt in parallel. Default 1")
parser.add_argument('-C',
                    '--color',
                    choices=list(COLS_XL.keys()),
                    default='black',
                    help=f"Color of input latent. Only supported with sd-ft-mse and sdxl VAEs. Default black, can be one of {list(COLS_XL.keys())}")
parser.add_argument('-c', '--color-scale', type=float, default=0.0, help="Alpha of colored latent. Default 0.0")
parser.add_argument('-m',
                    '--model',
                    type=str,
                    default=mdefault,
                    help=f"Huggingface model or Stable Diffusion safetensors checkpoint to load. Default {mdefault}")
parser.add_argument('-o', '--out', type=Path, default="/tmp/quickdif/", help=f"Output directory for images. Default {outdefault}")
parser.add_argument('-d', '--dtype', choices=dtypes, default="fp16", help=f"Data format for inference. Default fp16, can be one of {dtypes}")
parser.add_argument('--seed', type=int, nargs='*', help="Seed for deterministic outputs. If not set, will be random")
parser.add_argument('-S', '--sampler', choices=samplers, help=f"Override model's default sampler. Can be one of {samplers}")
parser.add_argument('--offload', choices=offload, default="model", help=f"Set amount of CPU offload. Can be one of {offload}. Default 'model'")
parser.add_argument('--compile', action='store_true', help="Compile unet with torch.compile()")
parser.add_argument('--no-trail', action='store_true', help="Do not force trailing timestep spacing. Changes seeds.")
parser.add_argument('--help', action='help')

args = parser.parse_args()

if args.out.is_dir():
    pass
elif not args.out.exists():
    args.out.mkdir()
else:
    raise ValueError("out must be directory")
# CLI }}}

# TORCH {{{
import torch, gc, inspect
from diffusers import (
    AutoPipelineForText2Image,
    # DiffusionPipeline, maybe re-add once multi-stage is manually implemented
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
import transformers
# from transformers import transformers.models.clip.tokenization_clip.CLIPTokenizer
from compel import Compel, ReturnedEmbeddingsType

torch.set_float32_matmul_precision('high')
AMD = 'AMD' in torch.cuda.get_device_name()
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
# TORCH }}}

# PIPE {{{
pipe_args = {
    "add_watermarker": False,
    "safety_checker": None,
    "torch_dtype": dtype,
    "use_safetensors": True,
    "watermarker": None,
}

if args.model.endswith('.safetensors'):
    pipe = StableDiffusionPipeline.from_single_file(args.model, **pipe_args)
else:
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, **pipe_args)

XL = isinstance(pipe, StableDiffusionXLPipeline)
SD = isinstance(pipe, StableDiffusionPipeline)

pipe.safety_checker = None
pipe.watermarker = None
if args.offload == "model": pipe.enable_model_cpu_offload()
elif args.offload == "sequential": pipe.enable_sequential_cpu_offload()
else: pipe.to('cuda')
# PIPE }}}

# MODEL {{{
weights = None
if hasattr(pipe, 'unet'): weights = pipe.unet
if hasattr(pipe, 'transformer'): weights = pipe.transformer

if args.compile:
    if weights: weights = torch.compile(weights)
elif AMD:
    if weights and hasattr(weights, 'set_default_attn_processor'): weights.set_default_attn_processor()
# MODEL }}}

# TOKENIZER/COMPEL {{{
if hasattr(pipe, 'tokenizer') and isinstance(pipe.tokenizer, transformers.models.clip.tokenization_clip.CLIPTokenizer):
    if XL:
        compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                        truncate_long_prompts=False)
    else:
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False)
# TOKENIZER/COMPEL }}}

# VAE {{{
if hasattr(pipe, 'vae'):
    pipe.vae.enable_slicing()
    if dtype != torch.float16: pipe.vae.force_upcast = False
    if AMD: pipe.vae.set_default_attn_processor()
# VAE }}}

# SCHEDULER {{{
if hasattr(pipe, 'scheduler'):
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
                args = list(args)
                num_inference_steps = args[0]
                num_inference_steps -= 1
                args[0] = num_inference_steps
            tsbase(*args, **kwargs)
            pipe.scheduler.timesteps = torch.cat([pipe.scheduler.timesteps, pipe.scheduler.timesteps[-1].remainder(num_inference_steps).reshape(1)])

        pipe.scheduler.set_timesteps = tspatch
    elif args.sampler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # what most UIs use
    if not args.no_trail: pipe.scheduler.config.timestep_spacing = 'trailing'
# SCHEDULER }}}

# INPUT TENSOR {{{
if hasattr(pipe, 'vae') and weights:
    size = [
        1, weights.config.in_channels, args.height // pipe.vae_scale_factor if args.height else weights.config.sample_size,
        args.width // pipe.vae_scale_factor if args.width else weights.config.sample_size
    ]

    # colored latents
    if args.color_scale > 0 and (SD or XL):
        sigma = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing').init_noise_sigma
        latent_input = torch.tensor(COLS_XL[args.color] if XL else COLS_FTMSE[args.color], dtype=dtype,
                                    device='cpu').mul(args.color_scale).div(sigma).expand([size[0], size[2], size[3], size[1]]).permute(
                                        (0, 3, 1, 2)).clone()
    else:
        latent_input = torch.zeros(size, dtype=dtype, device='cpu')

    size[0] = args.batch_size
# INPUT TENSOR }}}

# INPUT ARGS {{{
i32max = 2**31 - 1
seeds = [torch.randint(high=i32max, low=-i32max, size=(1, )).item()] if not args.seed else args.seed
key_dicts = [{"num_images_per_prompt": args.batch_size, 'seed': s + n * args.batch_size} for n in range(args.batch_count) for s in seeds]
key_dicts = [k | {'prompt': p} for k in key_dicts for p in args.prompts]
key_dicts = [k | {"negative_prompt": n} for k in key_dicts for n in args.negative]
if args.steps: key_dicts = [k | {'num_inference_steps': s} for k in key_dicts for s in args.steps]
if args.cfg: key_dicts = [k | {'guidance_scale': g, 'prior_guidance_scale': g, 'decoder_guidance_scale': g} for k in key_dicts for g in args.cfg]
if args.rescale: key_dicts = [k | {'guidance_rescale': g} for k in key_dicts for g in args.rescale]
# INPUT ARGS }}}

# INFERENCE {{{
print(f"Generating {len(key_dicts)} batches of {args.batch_size} images for {len(key_dicts) * args.batch_size} total...")
n = 0
for kwargs in key_dicts:
    seed = kwargs.pop('seed')
    if args.width: kwargs['width'] = args.width
    if args.height: kwargs['height'] = args.height
    print(kwargs)
    kwargs |= {
        "clean_caption": False,  # stop IF nag. what does this even do
    }

    # NOISE {{{
    if 'latent_input' in locals():
        latents = latent_input.expand(size).clone()
        seeds = []
        for (ln, latent) in enumerate(latents):
            seeds.append(seed + ln)
            generator = torch.manual_seed(seed + ln)
            # f32 noise for equal seeds amongst other UIs
            latent += torch.randn(latents.shape[1:], generator=generator, dtype=torch.float32)
        kwargs["latents"] = latents
        print("seeds:", ' '.join(map(str, seeds)))
    else:  # No input tensors for diffusion pipeline call?
        kwargs["generator"] = torch.manual_seed(seed)
        print("seed:", seed)
    # NOISE }}}

    # CONDITIONING {{{
    if 'compel' in locals():
        pos = kwargs.pop('prompt')
        neg = kwargs.pop('negative_prompt')
        if XL:
            ncond, npool = compel.build_conditioning_tensor(neg)
            pcond, ppool = compel.build_conditioning_tensor(pos)
            kwargs = kwargs | {'pooled_prompt_embeds': ppool, 'negative_pooled_prompt_embeds': npool}
        else:
            pcond = compel.build_conditioning_tensor(pos)
            ncond = compel.build_conditioning_tensor(neg)
        pcond, ncond = compel.pad_conditioning_tensors_to_same_length([pcond, ncond])
        kwargs |= {'prompt_embeds': pcond, 'negative_prompt_embeds': ncond}
    # CONDITIONING }}}

    # PROCESS {{{
    # make sure call doesnt err
    params = inspect.signature(pipe).parameters
    for k in list(kwargs.keys()):
        if k not in params: del kwargs[k]

    for image in pipe(**kwargs).images:
        p = args.out.joinpath(f"{n:05}.png")
        while p.exists():
            n += 1
            p = args.out.joinpath(f"{n:05}.png")

        image.save(p, format="PNG")
        del image, p
    # PROCESS }}}

    # CLEANUP {{{
    for k in ['kwargs', 'pcond', 'ncond', 'ppool', 'npool', 'latents', 'generator']:
        if k in locals():
            del locals()[k]
    if (lambda f, t: f / t)(*torch.cuda.mem_get_info()) < 0.25:
        gc.collect()
        torch.cuda.empty_cache()
    # CLEANUP }}}
# INFERENCE }}}
