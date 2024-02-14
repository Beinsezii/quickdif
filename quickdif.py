# LATENT COLORS {{{
COLS_XL = {
    "black": [-2.8232, 0.5033, 0.3139, 0.3359],
    "white": [2.3501, 0.2248, 1.2127, -1.0597],
    "red": [-2.5614, -2.5784, 1.3915, -1.6186],
    "green": [-0.4599, 1.8333, 3.4502, 1.1301],
    "blue": [0.0593, 2.1290, -2.3017, 0.5399],
    "cyan": [1.6195, 3.3881, 0.5599, 1.0360],
    "magenta": [-0.1252, -0.6654, -1.5711, -1.1750],
    "yellow": [-0.8608, -1.3759, 4.2304, -1.0693],
}
COLS_FTMSE = {
    "black": [-0.9953, -2.6024, 1.1153, 1.2966],
    "white": [2.1749, 1.4434, -0.0318, -1.1621],
    "red": [1.2575, -0.8768, -1.8788, 0.7792],
    "green": [0.7681, 0.8218, 2.3520, 1.8135],
    "blue": [-0.6330, -3.0466, 0.9888, -1.4022],
    "cyan": [0.5828, -0.1041, 2.9347, -0.0978],
    "magenta": [0.7387, -1.5557, -1.3593, -1.4386],
    "yellow": [2.3476, 2.0031, -0.0791, 1.6609],
}
# LATENT COLORS }}}

# CLI {{{
import argparse, os, inspect
from pathlib import Path
from PIL import Image
from math import ceil

outdefault = "/tmp/" if os.path.exists("/tmp/") else "./output/"
mdefault = "stabilityai/stable-diffusion-xl-base-1.0"
samplers = ["dpm", "dpmk", "ddim", "ddpm", "euler", "eulerk", "eulera"]
dtypes = ["fp16", "bf16", "fp32"]
offload = ["model", "sequential"]
noise_types = ["cpu16", "cpu16b", "cpu32", "cuda16", "cuda16b", "cuda32"]

parser = argparse.ArgumentParser(
    description="Quick and easy inference for a variety of Diffusers models. Not all models support all options", add_help=False
)
parser.add_argument("prompts", nargs="+", type=str)
parser.add_argument("-n", "--negative", type=str, nargs="*", default=["blurry"], help="Universal negative for all prompts. Default 'blurry'")
parser.add_argument("-w", "--width", type=int, help="Final output width. Default varies by model")
parser.add_argument("-h", "--height", type=int, help="Final output height. Default varies by model")
parser.add_argument("-s", "--steps", type=int, nargs="*", default=[30], help="Number of inference steps. Default 30. Can be unset")
parser.add_argument("-g", "--cfg", type=float, nargs="*", default=[6], help="Guidance for conditioning. Default 6. Can be unset")
parser.add_argument("-G", "--rescale", type=float, nargs="*", default=[0.7], help="Guidance rescale factor. Default 0.7. Can be unset")
parser.add_argument("-b", "--batch-count", type=int, default=1, help="Amount of times to run each prompt sequentially. Default 1")
parser.add_argument("-B", "--batch-size", type=int, default=1, help="Amount of times to run each prompt in parallel. Default 1")
parser.add_argument(
    "-C",
    "--color",
    choices=list(COLS_XL.keys()),
    default="black",
    help=f"Color of input latent. Only supported with sd-ft-mse and sdxl VAEs. Default black, can be one of {list(COLS_XL.keys())}",
)
parser.add_argument("-c", "--color-scale", type=float, default=0.0, help="Alpha of colored latent. Default 0.0")
parser.add_argument(
    "-m", "--model", type=str, default=mdefault, help=f"Huggingface model or Stable Diffusion safetensors checkpoint to load. Default {mdefault}"
)
parser.add_argument("-l", "--lora", type=str, help="Apply a Lora")
parser.add_argument("-L", "--lora-scale", type=float, default=1.0, help="Strength of the lora. Default 1.0")
parser.add_argument("-i", "--input", type=argparse.FileType(mode="rb"), help="Input image")
parser.add_argument("-d", "--denoise", type=float, default=1.0, help="Denoise amount. Default 1.0")
parser.add_argument("-o", "--out", type=Path, default="/tmp/quickdif/", help=f"Output directory for images. Default {outdefault}")
parser.add_argument("-D", "--dtype", choices=dtypes, default="fp16", help=f"Data format for inference. Default fp16, can be one of {dtypes}")
parser.add_argument("--seed", type=int, nargs="*", help="Seed for deterministic outputs. If not set, will be random")
parser.add_argument("-S", "--sampler", choices=samplers, help=f"Override model's default sampler. Can be one of {samplers}")
parser.add_argument(
    "--noise-type",
    choices=noise_types,
    default="cpu32",
    help=f"Device/precision for random noise if supported by pipeline. Can be one of {noise_types}. Default 'cpu32'",
)
parser.add_argument("--prior-steps-ratio", type=float, default=2.0, help="Ratio for prior/decoder steps. Default 2")
parser.add_argument("--offload", choices=offload, help=f"Set amount of CPU offload. Can be one of {offload}")
parser.add_argument("--compile", action="store_true", help="Compile unet with torch.compile()")
parser.add_argument("--no-trail", action="store_true", help="Do not force trailing timestep spacing. Changes seeds.")
parser.add_argument("--help", action="help")

args = parser.parse_args()

if args.out.is_dir():
    pass
elif not args.out.exists():
    args.out.mkdir()
else:
    raise ValueError("out must be directory")

input_image = None
if args.input:
    input_image = Image.open(args.input)

# start meta
base_meta = {"model": args.model, "noise_type": args.noise_type, "denoise": args.denoise, "url": "https://github.com/Beinsezii/quickdif"}
if args.batch_size > 1:
    base_meta["batch_size"] = args.batch_size
if args.sampler:
    base_meta["sampler"] = args.sampler
# CLI }}}

# TORCH {{{
import torch, transformers, tqdm
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableCascadeCombinedPipeline,
    StableCascadePriorPipeline,
    StableCascadeDecoderPipeline,
    DDIMScheduler,
    DDPMScheduler,
    # DiffusionPipeline, maybe re-add once multi-stage is manually implemented
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PixArtAlphaPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from compel import Compel, ReturnedEmbeddingsType
from PIL.PngImagePlugin import PngInfo

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
AMD = "AMD" in torch.cuda.get_device_name()
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
noise_dtype, noise_device = {
    "cpu16": (torch.float16, "cpu"),
    "cpu16b": (torch.bfloat16, "cpu"),
    "cpu32": (torch.float32, "cpu"),
    "cuda16": (torch.float16, "cuda"),
    "cuda16b": (torch.bfloat16, "cuda"),
    "cuda32": (torch.float32, "cuda"),
}[args.noise_type]
# TORCH }}}

# PIPE {{{
pipe_args = {
    "add_watermarker": False,
    "safety_checker": None,
    "torch_dtype": dtype,
    "use_safetensors": True,
    "watermarker": None,
}

if "cascade" in args.model:
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", **pipe_args).to("cuda")
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", **pipe_args).to("cuda")
    pipe = StableCascadeCombinedPipeline(
        decoder.tokenizer,
        decoder.text_encoder,
        decoder.decoder,
        decoder.scheduler,
        decoder.vqgan,
        prior.prior,
        prior.scheduler,
        # prior.feature_extractor,
        # prior.image_encoder,
    )
    del prior, decoder
elif args.model.endswith(".safetensors"):
    if input_image is not None:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(args.model, **pipe_args)
    else:
        pipe = StableDiffusionPipeline.from_single_file(args.model, **pipe_args)
else:
    if input_image is not None:
        pipe = AutoPipelineForImage2Image.from_pretrained(args.model, **pipe_args)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, **pipe_args)

XL = isinstance(pipe, StableDiffusionXLPipeline) or isinstance(pipe, StableDiffusionXLImg2ImgPipeline)
SD = isinstance(pipe, StableDiffusionPipeline) or isinstance(pipe, StableDiffusionImg2ImgPipeline)

pipe_params = inspect.signature(pipe).parameters

pipe.safety_checker = None
pipe.watermarker = None
if args.offload == "model":
    pipe.enable_model_cpu_offload()
elif args.offload == "sequential":
    pipe.enable_sequential_cpu_offload()
else:
    pipe.to("cuda")
# PIPE }}}

# MODEL {{{
weights = None

if hasattr(pipe, "unet"):
    if args.compile:
        pipe.unet = torch.compile(pipe.unet)
    weights = pipe.unet

if hasattr(pipe, "transformer"):
    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer)
    weights = pipe.transformer

if AMD and not args.compile and weights and hasattr(weights, "set_default_attn_processor"):
    weights.set_default_attn_processor()
# MODEL }}}

# LORA {{{
if args.lora_scale == 0:
    args.lora = None
if args.lora:
    pipe.load_lora_weights(args.lora)
    base_meta["lora"] = args.lora
    base_meta["lora_scale"] = args.lora_scale
# LORA }}}

# TOKENIZER/COMPEL {{{
if hasattr(pipe, "tokenizer") and isinstance(pipe.tokenizer, transformers.models.clip.tokenization_clip.CLIPTokenizer):
    if XL:
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
    else:
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False)
# TOKENIZER/COMPEL }}}

# VAE {{{
if hasattr(pipe, "vae"):
    pipe.vae.enable_slicing()
    if dtype != torch.float16:
        pipe.vae.force_upcast = False
    if AMD:
        pipe.vae.set_default_attn_processor()
# VAE }}}

# SCHEDULER {{{
if hasattr(pipe, "scheduler"):
    if args.sampler:
        pipe.scheduler = {
            "dpm": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=False),
            "dpmk": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True),
            "ddim": DDIMScheduler.from_config(pipe.scheduler.config, set_alpha_to_one=True),
            "ddpm": DDPMScheduler.from_config(pipe.scheduler.config),
            "euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
            "eulerk": EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True),
            "eulera": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        }[args.sampler]

    # what most UIs use
    if not args.no_trail:
        pipe.scheduler.config.timestep_spacing = "trailing"
        pipe.scheduler.config.steps_offset = 0
# SCHEDULER }}}

# INPUT TENSOR {{{
if hasattr(pipe, "vae") and weights and input_image is None:
    size = [
        args.batch_size,
        weights.config.in_channels,
        args.height // pipe.vae_scale_factor if args.height else weights.config.sample_size,
        args.width // pipe.vae_scale_factor if args.width else weights.config.sample_size,
    ]

    # colored latents
    cols = None
    if XL:
        cols = COLS_XL
    elif SD or isinstance(pipe, PixArtAlphaPipeline):
        cols = COLS_FTMSE
    if args.color_scale > 0 and cols:
        sigma = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing").init_noise_sigma
        latent_input = (
            torch.tensor(cols[args.color], dtype=dtype, device="cpu")
            .mul(args.color_scale)
            .div(sigma)
            .expand([size[0], size[2], size[3], size[1]])
            .permute((0, 3, 1, 2))
            .clone()
        )
        base_meta["color"] = args.color
        base_meta["color_scale"] = args.color_scale
    else:
        latent_input = torch.zeros(size, dtype=dtype, device="cpu")

elif hasattr(pipe, "vqgan") and hasattr(pipe, "prior_pipe") and input_image is None:
    if not args.height:
        args.height = 1024
    if not args.width:
        args.width = 1024
    size = [
        args.batch_size,
        pipe.prior_pipe.prior.config.c_in,
        ceil(args.height / pipe.prior_pipe.resolution_multiple),
        ceil(args.width / pipe.prior_pipe.resolution_multiple),
    ]
    latent_input = torch.zeros(size, dtype=dtype, device="cpu")
# INPUT TENSOR }}}

# INPUT ARGS {{{
base_dict = {
    "num_images_per_prompt": args.batch_size,
    "clean_caption": False,  # stop IF nag. what does this even do
    "strength": args.denoise,
}
if not args.negative:
    args.negative = [""]
if args.width:
    base_dict["width"] = args.width
if args.height:
    base_dict["height"] = args.height
if args.lora:
    base_dict["cross_attention_kwargs"] = {"scale": args.lora_scale}
i32max = 2**31 - 1
seeds = [torch.randint(high=i32max, low=-i32max, size=(1,)).item()] if not args.seed else args.seed
key_dicts = [base_dict | {"seed": s + n * args.batch_size} for n in range(args.batch_count) for s in seeds]
key_dicts = [k | {"prompt": p} for k in key_dicts for p in args.prompts]
key_dicts = [k | {"negative_prompt": n} for k in key_dicts for n in args.negative]
if args.steps:
    key_dicts = [
        k
        | (
            {"prior_num_inference_steps": s, "num_inference_steps": ceil(s // args.prior_steps_ratio)}
            if "prior_num_inference_steps" in pipe_params
            else {"num_inference_steps": s}
        )
        for k in key_dicts
        for s in args.steps
    ]
if args.cfg:
    key_dicts = [
        k
        | ({"guidance_scale": g, "prior_guidance_scale": g, "decoder_guidance_scale": 0.0 if isinstance(pipe, StableCascadeCombinedPipeline) else g})
        for k in key_dicts
        for g in args.cfg
    ]
if args.rescale:
    key_dicts = [k | {"guidance_rescale": g} for k in key_dicts for g in args.rescale]
# INPUT ARGS }}}

# INFERENCE {{{
print(f"Generating {len(key_dicts)} batches of {args.batch_size} images for {len(key_dicts) * args.batch_size} total...")
filenum = 0
if len(key_dicts) > 1:
    bar = tqdm.tqdm(desc="Images", total=len(key_dicts) * args.batch_size)
for kwargs in key_dicts:
    torch.cuda.empty_cache()
    meta = base_meta.copy()
    seed = kwargs.pop("seed")

    # NOISE {{{
    generators = [torch.Generator(noise_device).manual_seed(seed + n) for n in range(args.batch_size)]

    if "latent_input" in locals():
        latents = latent_input.clone()
        for latent, generator in zip(latents, generators):
            latent += torch.randn(latents.shape[1:], generator=generator, dtype=noise_dtype, device=noise_device).to("cpu")
        kwargs["latents"] = latents

    kwargs["generator"] = generators
    print("seeds:", " ".join([str(seed + n) for n in range(args.batch_size)]))
    # NOISE }}}

    # CONDITIONING {{{
    meta["prompt"] = kwargs["prompt"]
    meta["negative"] = kwargs["negative_prompt"]
    if "compel" in locals():
        pos = kwargs.pop("prompt")
        neg = kwargs.pop("negative_prompt")
        if XL:
            ncond, npool = compel.build_conditioning_tensor(neg)
            pcond, ppool = compel.build_conditioning_tensor(pos)
            kwargs = kwargs | {"pooled_prompt_embeds": ppool, "negative_pooled_prompt_embeds": npool}
        else:
            pcond = compel.build_conditioning_tensor(pos)
            ncond = compel.build_conditioning_tensor(neg)
        pcond, ncond = compel.pad_conditioning_tensors_to_same_length([pcond, ncond])
        kwargs |= {"prompt_embeds": pcond, "negative_prompt_embeds": ncond}
    # CONDITIONING }}}

    # PROCESS {{{
    # make sure call doesnt err
    if input_image is not None:
        kwargs["image"] = input_image
    for k in list(kwargs.keys()):
        if k not in pipe_params:
            del kwargs[k]

    if "num_inference_steps" in kwargs:
        meta["steps"] = kwargs["num_inference_steps"]
    if "guidance_scale" in kwargs:
        meta["cfg"] = kwargs["guidance_scale"]
    if "guidance_rescale" in kwargs:
        meta["rescale"] = kwargs["guidance_rescale"]
    for n, image in enumerate(pipe(**kwargs).images):
        p = args.out.joinpath(f"{filenum:05}.png")
        while p.exists():
            filenum += 1
            p = args.out.joinpath(f"{filenum:05}.png")
        pnginfo = PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        if "latents" in kwargs or n == 0:
            pnginfo.add_text("seed", str(seed + n))
        else:
            pnginfo.add_text("seed", f"{seed} + {n}")
        image.save(p, format="PNG", pnginfo=pnginfo)
    if len(key_dicts) > 1:
        bar.update(args.batch_size)
    # PROCESS }}}
# INFERENCE }}}
