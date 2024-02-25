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
import argparse, math
from inspect import signature
from pathlib import Path
from sys import exit

from PIL import Image, PngImagePlugin

samplers = [
    "dpm",
    "dpmk",
    "sdpm",
    "sdpmk",
    "dpm2",
    "dpm2k",
    "sdpm2",
    "sdpm2k",
    "dpm3",
    "dpm3k",
    "sdpm3",
    "sdpm3k",
    "ddim",
    "ddpm",
    "euler",
    "eulerk",
    "eulera",
]
dtypes = ["fp16", "bf16", "fp32"]
offload = ["model", "sequential"]
noise_types = ["cpu16", "cpu16b", "cpu32", "cuda16", "cuda16b", "cuda32"]

defaults = {
    "prompt": [""],
    "negative": ["blurry, noisy, cropped"],
    "steps": [30],
    "cfg": [6.0],
    "rescale": [0.7],
    "batch_count": 1,
    "batch_size": 1,
    "color": "black",
    "color_scale": 0.0,
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "denoise": 1.0,
    "out": Path("/tmp/quickdif/" if Path("/tmp/").exists() else "./output/"),
    "dtype": "fp16",
    "noise_type": "cpu32",
    "decoder_steps": -8,
}

parser = argparse.ArgumentParser(
    description="Quick and easy inference for a variety of Diffusers models. Not all models support all options", add_help=False
)
parser.add_argument("prompt", nargs="*", type=str)
parser.add_argument(
    "-n",
    "--negative",
    type=str,
    nargs="*",
    help="Universal negative for all prompts. Default 'blurry, noisy, cropped'",
)
parser.add_argument("-w", "--width", type=int, help="Final output width. Default varies by model")
parser.add_argument("-h", "--height", type=int, help="Final output height. Default varies by model")
parser.add_argument("-s", "--steps", type=int, nargs="*", help="Number of inference steps. Default 30. Can be unset")
parser.add_argument("-g", "--cfg", type=float, nargs="*", help="Guidance for conditioning. Default 6. Can be unset")
parser.add_argument("-G", "--rescale", type=float, nargs="*", help="Guidance rescale factor. Default 0.7. Can be unset")
parser.add_argument("-b", "--batch-count", type=int, help="Amount of times to run each prompt sequentially. Default 1")
parser.add_argument("-B", "--batch-size", type=int, help="Amount of times to run each prompt in parallel. Default 1")
parser.add_argument(
    "-C",
    "--color",
    choices=list(COLS_XL.keys()),
    help=f"Color of input latent. Only supported with sd-ft-mse and sdxl VAEs. Default black, can be one of {list(COLS_XL.keys())}",
)
parser.add_argument("-c", "--color-scale", type=float, help="Alpha of colored latent. Default 0.0")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help=f"Huggingface model or Stable Diffusion safetensors checkpoint to load. Default {defaults['model']}",
)
parser.add_argument("-l", "--lora", type=str, nargs="*", help="Apply Loras, ex. 'ms_paint.safetensors:::0.6'")
parser.add_argument(
    "-I",
    "--include",
    type=argparse.FileType(mode="rb"),
    help="Include parameters from another image. Only works with quickdif images",
)
parser.add_argument("-i", "--input", type=argparse.FileType(mode="rb"), help="Input image")
parser.add_argument("-d", "--denoise", type=float, help="Denoise amount. Default 1.0")
parser.add_argument("-o", "--out", type=Path, help=f"Output directory for images. Default {defaults['out']}")
parser.add_argument("-D", "--dtype", choices=dtypes, help=f"Data format for inference. Default fp16, can be one of {dtypes}")
parser.add_argument("--seed", type=int, nargs="*", help="Seed for deterministic outputs. If not set, will be random")
parser.add_argument("-S", "--sampler", choices=samplers, nargs="*", help=f"Override model's default sampler. Can be one of {samplers}")
parser.add_argument(
    "--noise-type",
    choices=noise_types,
    help=f"Device/precision for random noise if supported by pipeline. Can be one of {noise_types}. Default 'cpu32'",
)
parser.add_argument(
    "-ds", "--decoder-steps", type=int, help="Amount of steps for decoders. Default -8. If set to negative, uses quadratic slope √|s*ds|"
)
parser.add_argument("--offload", choices=offload, help=f"Set amount of CPU offload. Can be one of {offload}")
parser.add_argument("--comment", type=str, help="Add a comment to the image.")
parser.add_argument("--compile", action="store_true", help="Compile unet with torch.compile()")
parser.add_argument("--no-trail", action="store_true", help="Do not force trailing timestep spacing. Changes seeds.")
parser.add_argument("--no-xl-vae", action="store_true", help="Do not override the SDXL VAE.")
parser.add_argument("--print", action="store_true", help="Print out generation params and exit.")
parser.add_argument("--help", action="help")

args = parser.parse_args()
if len(args.prompt) == 0:  # positionals will always accumulate
    args.prompt = None

# include
include = {}
if args.include:
    with Image.open(args.include) as meta_image:
        include = {"width": meta_image.width, "height": meta_image.height}
        for k, v in meta_image.text.items():
            match k:
                case "prompt" | "negative" | "sampler":
                    include[k] = [v]
                case "cfg" | "rescale":
                    include[k] = [float(v)]
                case "steps" | "seed":
                    include[k] = [int(v)]
                case "model" | "noise_type" | "color":
                    include[k] = v
                case "denoise" | "color_scale":
                    include[k] = float(v)
                case "decoder_steps":
                    include[k] = int(v)
                case "lora":
                    include[k] = v.split("\x1f")
                case "url" | "batch_size" | "comment":
                    pass
                case other:
                    print(f'Unknown key "{other}: {v}"')

for k, v in (defaults | include).items():
    assert hasattr(args, k)
    if getattr(args, k, None) is None:
        setattr(args, k, v)

if args.print:
    print("\n".join([f"{k}: {v}" for k, v in vars(args).items()]))
    exit()

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
if args.comment:
    base_meta["comment"] = args.comment
# CLI }}}

# TORCH {{{
import torch, transformers, tqdm, signal
from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PixArtAlphaPipeline,
    StableCascadeCombinedPipeline,
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from compel import Compel, ReturnedEmbeddingsType


# elegent solution from <https://stackoverflow.com/questions/842557/>
class SmartSigint:
    def __init__(self, num=1, job_name=None):
        self.num = num
        self.job_name = job_name if job_name is not None else "current job"

    def __enter__(self):
        self.count = 0
        self.received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.received = (sig, frame)
        if self.count >= self.num:
            print(f"\nSIGINT received {self.count+1} times, forcibly aborting {self.job_name}")
            self.terminate()
        else:
            print(f"\nSIGINT received, waiting for {self.job_name} to complete before exiting.\nRequire {self.num - self.count} more to abort.")
        self.count += 1

    def terminate(self):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.received:
            self.old_handler(*self.received)

    def __exit__(self, _type, _value, _traceback):
        self.terminate()


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
AMD = "AMD" in torch.cuda.get_device_name()
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
if "cascade" in args.model and dtype == torch.float16:
    dtype = torch.bfloat16
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

with SmartSigint(num=2, job_name="model load"):
    if "cascade" in args.model:
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", **pipe_args)
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", **pipe_args)
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

pipe_params = signature(pipe).parameters

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
adapters = []

if args.lora:
    for n, lora in enumerate(args.lora):
        split = lora.rsplit(":::")
        path = split[0]
        scale = 1.0 if len(split) < 2 else float(split[1])
        if scale == 0.0:
            continue
        name = f"LA{n}"
        pipe.load_lora_weights(path, adapter_name=name)
        adapters.append(
            {
                "name": name,
                "path": path,
                "scale": scale,
            }
        )

if adapters:
    pipe.set_adapters(list(map(lambda a: a["name"], adapters)), list(map(lambda a: a["scale"], adapters)))
    pipe.fuse_lora(list(map(lambda a: a["name"], adapters)))
    # re-construct args without nulled loras
    base_meta["lora"] = "\x1f".join(map((lambda a: a["path"] if a["scale"] == 1.0 else f'{a["path"]}:::{a["scale"]}'), adapters))
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
    if XL and not args.no_xl_vae:
        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=pipe.vae.dtype, use_safetensors=True).to(pipe.vae.device)
    pipe.vae.enable_slicing()
    if dtype != torch.float16:
        pipe.vae.force_upcast = False
    if AMD:
        pipe.vae.set_default_attn_processor()
# VAE }}}

# SCHEDULER {{{
schedulers = None
if hasattr(pipe, "scheduler"):
    sampler_map = {
        "dpm": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=1, use_karras_sigmas=False),
        "dpmk": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=1, use_karras_sigmas=True),
        "sdpm": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=1, use_karras_sigmas=False
        ),
        "sdpmk": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=1, use_karras_sigmas=True
        ),
        "dpm2": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=False),
        "dpm2k": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True),
        "sdpm2": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=False
        ),
        "sdpm2k": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True
        ),
        "dpm3": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=3, use_karras_sigmas=False),
        "dpm3k": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=3, use_karras_sigmas=True),
        "sdpm3": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=3, use_karras_sigmas=False
        ),
        "sdpm3k": DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=3, use_karras_sigmas=True
        ),
        "ddim": DDIMScheduler.from_config(pipe.scheduler.config, set_alpha_to_one=True),
        "ddpm": DDPMScheduler.from_config(pipe.scheduler.config),
        "euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
        "eulerk": EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True),
        "eulera": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    }
    assert list(sampler_map.keys()) == samplers
    if args.sampler:
        schedulers = []
        for s in args.sampler:
            sampler = sampler_map[s]
            if not args.no_trail:
                sampler.config.timestep_spacing = "trailing"
                sampler.config.steps_offset = 0
            schedulers.append((s, sampler))

    if schedulers:
        if len(schedulers) == 1:  # consume single samplers
            name, sched = schedulers[0]
            pipe.scheduler = sched
            base_meta["sampler"] = name
            schedulers = None
    elif not args.no_trail:
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
        math.ceil(args.height / pipe.prior_pipe.config.resolution_multiple),
        math.ceil(args.width / pipe.prior_pipe.config.resolution_multiple),
    ]
    latent_input = torch.zeros(size, dtype=dtype, device="cpu")

else:
    print(f"Model {args.model} not able to use pre-noised latents.\nNoise type {args.noise_type} will not be respected.")
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
i32max = 2**31 - 1
seeds = [torch.randint(high=i32max, low=-i32max, size=(1,)).item()] if not args.seed else args.seed
key_dicts = [base_dict | {"seed": s + n * args.batch_size} for n in range(args.batch_count) for s in seeds]
key_dicts = [k | {"prompt": p} for k in key_dicts for p in args.prompt]
key_dicts = [k | {"negative_prompt": n} for k in key_dicts for n in args.negative]
if args.steps:
    key_dicts = [
        k
        | (
            {
                "prior_num_inference_steps": s,
                "num_inference_steps": round(math.sqrt(abs(s * args.decoder_steps))) if args.decoder_steps < 0 else args.decoder_steps,
            }
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

if schedulers:
    key_dicts = [k | {"scheduler": s} for k in key_dicts for s in schedulers]

# INPUT ARGS }}}

# INFERENCE {{{
print(f"Generating {len(key_dicts)} batches of {args.batch_size} images for {len(key_dicts) * args.batch_size} total...")
filenum = 0
if len(key_dicts) > 1:
    bar = tqdm.tqdm(desc="Images", total=len(key_dicts) * args.batch_size)
for kwargs in key_dicts:
    with SmartSigint(job_name="current batch"):
        torch.cuda.empty_cache()
        meta = base_meta.copy()
        seed = kwargs.pop("seed")

        if "scheduler" in kwargs:
            name, sched = kwargs.pop("scheduler")
            pipe.scheduler = sched
            meta["sampler"] = name

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

        if "prior_num_inference_steps" in kwargs:
            meta["steps"] = kwargs["prior_num_inference_steps"]
            meta["decoder_steps"] = args.decoder_steps
        elif "num_inference_steps" in kwargs:
            meta["steps"] = kwargs["num_inference_steps"]

        if "prior_guidance_scale" in kwargs:
            meta["cfg"] = kwargs["prior_guidance_scale"]
        elif "guidance_scale" in kwargs:
            meta["cfg"] = kwargs["guidance_scale"]

        if "guidance_rescale" in kwargs:
            meta["rescale"] = kwargs["guidance_rescale"]

        for n, image in enumerate(pipe(**kwargs).images):
            p = args.out.joinpath(f"{filenum:05}.png")
            while p.exists():
                filenum += 1
                p = args.out.joinpath(f"{filenum:05}.png")
            pnginfo = PngImagePlugin.PngInfo()
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
