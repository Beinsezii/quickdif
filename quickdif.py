import argparse
import enum
import functools
import json
import math
import random
import signal
from copy import copy
from inspect import signature
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from sys import exit
from typing import Any, List, Tuple

import numpy as np
import tomllib
import tqdm
from PIL import Image, PngImagePlugin


# FUNCTIONS {{{
# pexpand {{{
@functools.cache
def _pexpand_bounds(string: str, body: Tuple[str, str]) -> None | Tuple[int, int]:
    start = len(string) + 1
    end = 0
    escape = False
    for n, c in enumerate(string):
        if escape:
            escape = False
            continue
        elif c == "\\":
            escape = True
            continue
        elif c == body[0]:
            start = n
        elif c == body[1]:
            end = n
        if end > start:
            return (start, end)
    return None


def _pexpand(prompt: str, body: Tuple[str, str] = ("{", "}"), sep: str = "|", single: bool = False) -> List[str]:
    bounds = _pexpand_bounds(prompt, body)
    # Split first body; first close after last open
    if bounds is not None:
        prefix = prompt[: bounds[0]]
        suffix = prompt[bounds[1] + 1 :]
        values = []
        current = ""
        escape = False
        for c in prompt[bounds[0] + 1 : bounds[1]]:
            if escape:
                current += c
                escape = False
                continue
            elif c == "\\":
                escape = True
            if c == sep and not escape:
                values.append(current)
                current = ""
            else:
                current += c
        values.append(current)
        if single:
            values = [random.choice(values)]
        results = [prefix + v + suffix for v in values]
    else:
        results = [prompt]

    # Recurse on unexpanded bodies
    results, iter = [], results
    for result in iter:
        if _pexpand_bounds(result, body) is None:
            results.append(result)
        else:
            results += pexpand(result, body, sep, single)

    if single:
        results = [random.choice(results)]

    results[:] = dict.fromkeys(results)

    return [result.replace("\\\\", "\x1a").replace("\\", "").replace("\x1a", "\\") for result in results]


@functools.cache
def _pexpand_cache(*args, **kwargs):
    return _pexpand(*args, **kwargs)


def pexpand(prompt: str, body: Tuple[str, str] = ("{", "}"), sep: str = "|", single: bool = False) -> List[str]:
    if single:
        return _pexpand(prompt, body, sep, single)
    else:
        return _pexpand_cache(prompt, body, sep, single)


# }}}
def oversample(population: list, k: int):
    samples = []
    while len(samples) < k:
        samples += random.sample(population, min(len(population), k - len(samples)))
    assert len(samples) == k
    return samples


# }}}

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

# QDPARAMS {{{


# Enums {{{
@enum.unique
class Iter(enum.StrEnum):
    Basic = enum.auto()
    Walk = enum.auto()
    Shuffle = enum.auto()


@enum.unique
class Sampler(enum.StrEnum):
    Default = enum.auto()
    Ddim = enum.auto()
    Ddpm = enum.auto()
    Euler = enum.auto()
    EulerK = enum.auto()
    EulerA = enum.auto()
    Dpm = enum.auto()
    DpmK = enum.auto()
    SDpm = enum.auto()
    SDpmK = enum.auto()
    Dpm2 = enum.auto()
    Dpm2K = enum.auto()
    SDpm2 = enum.auto()
    SDpm2K = enum.auto()
    Dpm3 = enum.auto()
    Dpm3K = enum.auto()
    # SDpm3 = enum.auto()
    # SDpm3K = enum.auto()


@enum.unique
class Spacing(enum.StrEnum):
    Leading = enum.auto()
    Trailing = enum.auto()
    Linspace = enum.auto()


@enum.unique
class DType(enum.StrEnum):
    FP16 = enum.auto()
    BF16 = enum.auto()
    FP32 = enum.auto()


@enum.unique
class Offload(enum.StrEnum):
    NONE = enum.auto()  # why no assign to None?
    Model = enum.auto()
    Sequential = enum.auto()


@enum.unique
class NoiseType(enum.StrEnum):
    Cpu16 = enum.auto()
    Cpu16B = enum.auto()
    Cpu32 = enum.auto()
    Cuda16 = enum.auto()
    Cuda16B = enum.auto()
    Cuda32 = enum.auto()


@enum.unique
class Attention(enum.StrEnum):
    Default = enum.auto()
    Sdp = enum.auto()
    SubQuad = enum.auto()
    RocmFlash = enum.auto()


@enum.unique
class LatentColor(enum.StrEnum):
    Red = enum.auto()
    Green = enum.auto()
    Blue = enum.auto()
    Cyan = enum.auto()
    Magenta = enum.auto()
    Yellow = enum.auto()
    Black = enum.auto()
    White = enum.auto()


class QDParam:
    def __init__(
        self,
        name: str,
        typing: type,
        value: Any = None,
        short: str | None = None,
        long: str | None = None,
        help: str | None = None,
        multi: bool = False,
        meta: bool = False,
    ):
        self.name = name
        self.typing = typing
        self.help = help
        self.short = short
        self.long = long
        self.multi = multi
        self.meta = meta

        self.value = value
        self.default = copy(self.value)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new):
        if isinstance(new, list):
            if len(new) == 0:
                new = None
        if new is None:
            self._value = new
        elif isinstance(new, list):
            if self.multi:
                new = [self.typing(v) for v in new]
                self._value = new
            else:
                raise ValueError(f"Refusing to assign list '{new}' to non-multi QDParam '{self.name}'")
        else:
            if self.multi:
                self._value = [self.typing(new)]
            else:
                self._value = self.typing(new)


# }}}


# Parameters {{{
params = [
    ### Batching
    QDParam("prompt", str, multi=True, meta=True, help="Positive prompt"),
    QDParam("negative", str, short="-n", long="--negative", multi=True, meta=True, help="Negative prompt"),
    QDParam("seed", int, short="-e", long="--seed", multi=True, meta=True, help="Seed for RNG"),
    QDParam(
        "steps",
        int,
        short="-s",
        long="--steps",
        value=30,
        multi=True,
        meta=True,
        help="Amount of denoising steps. Prior/Decoder models this only affects the Prior",
    ),
    QDParam(
        "decoder_steps",
        int,
        short="-ds",
        long="--decoder-steps",
        value=-8,
        multi=True,
        meta=True,
        help="Amount of denoising steps for the Decoder if applicable",
    ),
    QDParam(
        "guidance",
        float,
        short="-g",
        long="--guidance",
        value=5.0,
        multi=True,
        meta=True,
        help="CFG/Classier-Free Guidance. Will guide diffusion more strongly towards the prompts. High values will produce unnatural images",
    ),
    QDParam(
        "decoder_guidance", float, short="-dg", long="--decoder-guidance", multi=True, meta=True, help="Guidance for the Decoder stage if applicable"
    ),
    QDParam(
        "rescale",
        float,
        short="-G",
        long="--rescale",
        value=0.0,
        multi=True,
        meta=True,
        help="Rescale the noise during guidance. Moderate values may help produce more natural images when using strong guidance",
    ),
    QDParam(
        "denoise", float, short="-d", long="--denoise", multi=True, meta=True, help="Denoising amount for Img2Img. Higher values will change more"
    ),
    QDParam(
        "noise_type",
        NoiseType,
        short="-nt",
        long="--noise-type",
        value=NoiseType.Cpu32,
        multi=True,
        meta=True,
        help="Device and precision to source RNG from. To reproduce seeds from other diffusion programs it may be necessary to change this",
    ),
    QDParam(
        "noise_power",
        float,
        short="-np",
        long="--noise-power",
        multi=True,
        meta=True,
        help="Multiplier to the initial latent noise if applicable. <1 for smoother, >1 for more details",
    ),
    QDParam(
        "color",
        LatentColor,
        short="-C",
        long="--color",
        value=LatentColor.Black,
        multi=True,
        meta=True,
        help="Color of initial latent noise if applicable. Currently only for XL and SD-FT-MSE latent spaces",
    ),
    QDParam("color_power", float, short="-c", long="--color-power", multi=True, meta=True, help="Power/opacity of colored latent noise"),
    QDParam(
        "variance_scale",
        int,
        short="-vs",
        long="--variance-scale",
        value=2,
        multi=True,
        meta=True,
        help="Amount of 'zones' for variance noise. '2' will make a 2x2 grid or 4 tiles",
    ),
    QDParam(
        "variance_power",
        float,
        short="-vp",
        long="--variance-power",
        multi=True,
        meta=True,
        help="Power/opacity for variance noise. Variance noise simply adds randomly generated colored zones to encourage new compositions on overfitted models",
    ),
    QDParam("pixelate", float, long="--pixelate", multi=True, meta=True, help="Pixelate image using a divisor. Best used with a pixel art Lora"),
    QDParam("posterize", int, long="--posterize", multi=True, meta=True, help="Set amount of colors per channel. Best used with --pixelate"),
    QDParam(
        "sampler",
        Sampler,
        short="-S",
        long="--sampler",
        value=Sampler.Default,
        multi=True,
        meta=True,
        help="""Sampler to use in denoising. Naming scheme is as follows:
euler/ddim/ddpm - Literal names;
dpm - DPM++;
k - Use karras sigmas;
s - Use SDE stochastic noise;
a - Use ancestral sampling;
2/3 - Use 2nd/3rd order sampling;
Ex. 'sdpm2k' is equivalent to 'DPM++ 2M SDE Karras'""",
    ),
    QDParam("spacing", Spacing, long="--spacing", value=Spacing.Trailing, multi=True, meta=True, help="Sampler timestep spacing"),
    ### Global
    QDParam("width", int, short="-w", long="--width", help="Output image width"),
    QDParam("height", int, short="-h", long="--height", help="Output image height"),
    QDParam(
        "model",
        str,
        short="-m",
        long="--model",
        value="stabilityai/stable-diffusion-xl-base-1.0",
        meta=True,
        help="Safetensor file or HuggingFace model ID",
    ),
    QDParam("lora", str, short="-l", long="--lora", meta=True, multi=True, help='Apply Loras, ex. "ms_paint.safetensors:::0.6"'),
    QDParam("batch_count", int, short="-b", long="--batch-count", value=1, help="Behavior dependant on 'iter'"),
    QDParam("batch_size", int, short="-B", long="--batch-size", value=1, help="Amount of images to produce in each job"),
    QDParam(
        "iter",
        Iter,
        long="--iter",
        value=Iter.Basic,
        help="""Controls how jobs are created:
'basic' - Run every combination of parameters 'batch_count' times, incrementing seed each 'batch_count';
'walk' - Run every combination of parameters 'batch_count' times, incrementing seed for every individual job;
'shuffle' - Pick randomly from all given parameters 'batch_count' times""",
    ),
    ### System
    QDParam(
        "output",
        Path,
        short="-o",
        long="--output",
        value=Path("/tmp/quickdif/" if Path("/tmp/").exists() else "./output/"),
        help="Output directory for images",
    ),
    QDParam(
        "dtype",
        DType,
        short="-dt",
        long="--dtype",
        value=DType.FP16,
        help="Data format for inference. Should be left at FP16 unless the device or model does not work properly",
    ),
    QDParam(
        "offload",
        Offload,
        long="--offload",
        value=Offload.NONE,
        help="Set amount of CPU offload. In most UIs, 'model' is equivalent to --med-vram while 'sequential' is equivalent to --low-vram",
    ),
    QDParam("attention", Attention, value=Attention.Default, long="--attention", help="Select attention processor to use"),
    QDParam("compile", bool, long="--compile", help="Compile network with torch.compile()"),
    QDParam("tile", bool, long="--tile", help="Tile VAE. Slicing is already used by default so only set tile if creating very large images"),
    QDParam("xl_vae", bool, long="--xl-vae", help="Override the SDXL VAE. Useful for models that use the broken 1.0 vae"),
    QDParam("disable_amd_patch", bool, long="--disable-amd-patch", help="Do not monkey patch the torch SDPA function on AMD cards"),
]
params = {param.name: param for param in params}
# }}}
# QDPARAMS }}}

# CLI {{{
parser = argparse.ArgumentParser(
    description="Quick and easy inference for a variety of Diffusers models. Not all models support all options", add_help=False
)
for param in params.values():
    args = [param.short] if param.short else []
    args.append(param.long if param.long else param.name)

    kwargs = {}

    help = [param.help]
    if isinstance(param.value, list) and len(param.value) == 1:
        help += [f'Default "{param.value[0]}"']
    elif param.value is not None:
        help += [f'Default "{param.value}"']
    help = ". ".join([h for h in help if h is not None])

    if help:
        kwargs["help"] = help

    if param.typing == bool and param.multi is False:
        kwargs["action"] = argparse.BooleanOptionalAction
    else:
        kwargs["type"] = param.typing

        if issubclass(param.typing, enum.Enum):
            kwargs["choices"] = [e.value for e in param.typing]

        if param.multi:
            kwargs["nargs"] = "*"
        else:
            kwargs["nargs"] = "?"

    parser.add_argument(*args, **kwargs)

parser.add_argument("-i", "--input", type=argparse.FileType(mode="rb"), help="Input image")
parser.add_argument(
    "-I", "--include", type=argparse.FileType(mode="rb"), nargs="*", help="Include parameters from another image. Only works with quickdif images"
)
parser.add_argument("--json", type=argparse.FileType(mode="a+b"), help="Output settings to JSON")
# It would be nice to write toml but I don't think its worth a 3rd party lib
# parser.add_argument("--toml", type=argparse.FileType(mode="wb"), help="Output settings to TOML")
parser.add_argument("--comment", type=str, help="Add a comment to the image.")
parser.add_argument("--print", action="store_true", help="Print out generation params and exit.")
parser.add_argument("--help", action="help")

args = vars(parser.parse_args())

for ext in "json", "toml":
    default = Path(__file__).parent.joinpath(Path(f"quickdif.{ext}"))
    if default.exists():
        reader = default.open(mode="rb")
        if args["include"] is None:
            args["include"] = [reader]
        else:
            args["include"].insert(0, reader)

if args["include"]:
    for reader in args["include"]:
        data = reader.read()
        try:
            data = data.decode()
            for f, e in [
                (tomllib.loads, tomllib.TOMLDecodeError),
                (json.loads, json.JSONDecodeError),
            ]:
                try:
                    for k, v in f(data).items():
                        if k in params:
                            try:
                                params[k].value = v
                            except ValueError:
                                print(f"Config value '{v}' cannot be assigned to parameter '{params[k].name}', ignoring")
                        else:
                            print(f'Unknown key in serial config "{k}"')
                    break
                except e:
                    pass
        except UnicodeDecodeError:
            with Image.open(BytesIO(data)) as meta_image:
                params["width"].value = meta_image.width
                params["height"].value = meta_image.height
                for k, v in getattr(meta_image, "text", {}).items():
                    if k in params:
                        if k == "lora":
                            params[k].value = v.split("\x1f")
                        elif params[k].meta:
                            try:
                                params[k].value = v
                            except ValueError:
                                print(f"Config value '{v}' cannot be assigned to parameter '{params[k].name}', ignoring")

for id, val in args.items():
    if id in params and val is not None and not (isinstance(val, list) and len(val) == 0 and params[id].long is None and params[id].short is None):
        params[id].value = val

args = {k: v for k, v in args.items() if k not in params}

if args.get("json", None) is not None:
    dump = {}
    for k, v in params.items():
        if v.value != v.default:
            v = v.value
            if isinstance(v, Path):
                v = str(v)
            dump[k] = v
    s = json.dumps(dump)
    try:
        args["json"].seek(0)
        args["json"].truncate()
    except UnsupportedOperation:
        pass
    args["json"].write(s.encode())
    exit()

for key in "prompt", "negative":
    if params[key].value is not None:
        params[key].value = [expanded for nested in [pexpand(p) for p in params[key].value] for expanded in nested]

if args.get("print", False):
    print("\n".join([f"{p.name}: {p.value}" for p in params.values()]))
    exit()

if params["output"].value.is_dir():
    pass
elif not params["output"].value.exists():
    params["output"].value.mkdir()
else:
    raise ValueError("out must be directory")

input_image = None
if args["input"]:
    input_image = Image.open(args["input"])

base_meta = {"model": params["model"].value, "url": "https://github.com/Beinsezii/quickdif"}

if args.get("comment", ""):
    try:
        with open(args["comment"], "r") as f:
            base_meta["comment"] = f.read()
    except Exception:
        base_meta["comment"] = args["comment"]
# CLI }}}

# TORCH {{{
# Load Torch and libs that depend on it after the CLI cause it's laggy.
import torch  # noqa: E402

amd_hijack = False
if not params["disable_amd_patch"].value:
    if "AMD" in torch.cuda.get_device_name() or "Radeon" in torch.cuda.get_device_name():
        try:
            from flash_attn import flash_attn_func

            sdpa = torch.nn.functional.scaled_dot_product_attention

            def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                if query.shape[3] <= 128 and attn_mask is None:
                    result = flash_attn_func(
                        q=query.transpose(1, 2),
                        k=key.transpose(1, 2),
                        v=value.transpose(1, 2),
                        dropout_p=dropout_p,
                        causal=is_causal,
                        softmax_scale=scale,
                    )
                    hidden_states = result.transpose(1, 2) if result is not None else None
                else:
                    hidden_states = sdpa(
                        query=query,
                        key=key,
                        value=value,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )
                return hidden_states

            torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
            amd_hijack = True
            print("# # #\nHijacked SDPA with ROCm Flash Attention\n# # #")
        except ImportError as e:
            print(f"# # #\nCould not load Flash Attention for hijack:\n{e}\n# # #")
    else:
        print(f"# # #\nCould not detect AMD GPU from:\n{torch.cuda.get_device_name()}\n# # #")

import transformers  # noqa: E402
from compel import Compel, ReturnedEmbeddingsType  # noqa: E402

from diffusers import (  # noqa: E402
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
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: E402

try:
    from attn_custom import SubQuadraticCrossAttnProcessor as subquad_processor
except ImportError:
    subquad_processor = None

try:
    from attn_custom import FlashAttnProcessor as rocm_flash_processor
except ImportError:
    rocm_flash_processor = None


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
        if self.received and callable(self.old_handler):
            self.old_handler(*self.received)

    def __exit__(self, *_):
        self.terminate()


torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
dtype = {DType.FP16: torch.float16, DType.BF16: torch.bfloat16, DType.FP32: torch.float32}[params["dtype"].value]
if "cascade" in params["model"].value and dtype == torch.float16:
    dtype = torch.bfloat16
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
    if "stabilityai/stable-cascade" in params["model"].value:
        pipe = StableCascadeCombinedPipeline.from_pretrained(params["model"].value, **pipe_args)
    elif params["model"].value.endswith(".safetensors"):
        if input_image is not None:
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(params["model"].value, **pipe_args)
            except:  # noqa: E722
                pipe = StableDiffusionImg2ImgPipeline.from_single_file(params["model"].value, **pipe_args)
        else:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(params["model"].value, **pipe_args)
            except:  # noqa: E722
                pipe = StableDiffusionPipeline.from_single_file(params["model"].value, **pipe_args)
    else:
        if input_image is not None:
            pipe = AutoPipelineForImage2Image.from_pretrained(params["model"].value, **pipe_args)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(params["model"].value, **pipe_args)

XL = isinstance(pipe, StableDiffusionXLPipeline) or isinstance(pipe, StableDiffusionXLImg2ImgPipeline)
SD = isinstance(pipe, StableDiffusionPipeline) or isinstance(pipe, StableDiffusionImg2ImgPipeline)

assert callable(pipe)
pipe_params = signature(pipe).parameters

pipe.safety_checker = None
pipe.watermarker = None
match params["offload"].value:
    case Offload.NONE:
        pipe = pipe.to("cuda")
    case Offload.Model:
        pipe.enable_model_cpu_offload()
    case Offload.Sequential:
        pipe.enable_sequential_cpu_offload()
    case other:
        raise ValueError
# PIPE }}}

# ATTENTION {{{
if not params["compile"].value:
    processor = None
    match params["attention"].value:
        case Attention.Sdp:
            processor = AttnProcessor2_0()
        case Attention.Default:
            pass
        case Attention.SubQuad:
            if subquad_processor is not None:
                processor = subquad_processor(query_chunk_size=2**12, kv_chunk_size=2**15)
            else:
                print('\nAttention Processor "subquad" not available.\n')
        case Attention.RocmFlash:
            if amd_hijack:
                print('\nIgnoring attention procesor "rocm_flash" as SDPA was patched.\n')
            elif rocm_flash_processor is not None:
                processor = rocm_flash_processor()
            else:
                print('\nAttention Processor "rocm_flash" not available.\n')

    for id, item in [("pipe", pipe)] + [
        (id, getattr(pipe, id, None))
        for id in [
            "unet",
            "vae",
            "transformer",
            "prior",
            "prior_prior",
            "decoder",
            "vqgan",
        ]
    ]:
        if item is not None:
            if hasattr(item, "set_attn_processor") and processor is not None:
                item.set_attn_processor(processor)

# }}}

# COMPILE {{{
if params["compile"].value:
    if hasattr(pipe, "unet"):
        pipe.unet = torch.compile(pipe.unet)
    if hasattr(pipe, "transformer"):
        pipe.transformer = torch.compile(pipe.transformer)
# COMPILE }}}

# LORA {{{
adapters = []

if params["lora"].value:
    for n, lora in enumerate(params["lora"].value):
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
else:
    compel = None
# TOKENIZER/COMPEL }}}

# VAE {{{
if hasattr(pipe, "vae"):
    if XL and params["xl_vae"].value:
        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=pipe.vae.dtype, use_safetensors=True).to(pipe.vae.device)
    if params["tile"].value:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.enable_slicing()
    if dtype != torch.float16:
        pipe.vae.config.force_upcast = False
# VAE }}}

# SCHEDULER {{{
schedulers = None
if hasattr(pipe, "scheduler"):
    sampler_map = {
        Sampler.Default: pipe.scheduler,
        Sampler.Ddim: DDIMScheduler.from_config(pipe.scheduler.config),
        Sampler.Ddpm: DDPMScheduler.from_config(pipe.scheduler.config),
        Sampler.Euler: EulerDiscreteScheduler.from_config(pipe.scheduler.config),
        Sampler.EulerK: EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True),
        Sampler.EulerA: EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        Sampler.Dpm: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=1, use_karras_sigmas=False
        ),
        Sampler.DpmK: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=1, use_karras_sigmas=True
        ),
        Sampler.SDpm: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=1, use_karras_sigmas=False
        ),
        Sampler.SDpmK: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=1, use_karras_sigmas=True
        ),
        Sampler.Dpm2: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=False
        ),
        Sampler.Dpm2K: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True
        ),
        Sampler.SDpm2: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=False
        ),
        Sampler.SDpm2K: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True
        ),
        Sampler.Dpm3: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=3, use_karras_sigmas=False
        ),
        Sampler.Dpm3K: DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=3, use_karras_sigmas=True
        ),
        # Sampler.SDpm3: DPMSolverMultistepScheduler.from_config(
        #     pipe.scheduler.config, final_sigmas_type="zero", algorithm_type="sde-dpmsolver++", solver_order=3, use_karras_sigmas=False
        # ),
        # Sampler.SDpm3K: DPMSolverMultistepScheduler.from_config(
        #     pipe.scheduler.config, final_sigmas_type="zero", algorithm_type="sde-dpmsolver++", solver_order=3, use_karras_sigmas=True
        # ),
    }
    if params["sampler"].value:
        schedulers = []
        for s in params["sampler"].value:
            sampler = sampler_map[s]
            schedulers.append((s, sampler))

    if params["spacing"].value:
        schedulers = [
            (
                name,
                sched.from_config(
                    sched.config,
                    timestep_spacing=space,
                    steps_offset=0,
                    set_alpha_to_one=True,
                    final_sigmas_type="zero",
                ),
            )
            for name, sched in (schedulers if schedulers else [("default", pipe.scheduler)])
            for space in params["spacing"].value
        ]

    # consume single samplers
    if schedulers and len(schedulers) == 1:
        name, sched = schedulers[0]
        base_meta["sampler"] = name
        if params["spacing"].value and hasattr(sched.config, "timestep_spacing"):
            base_meta["spacing"] = sched.config.timestep_spacing
        pipe.scheduler = sched
        schedulers = None
# SCHEDULER }}}

# INPUT TENSOR {{{
size = None
if input_image is None:
    factor: float | None = None
    channels: int | None = None
    default_size: int | None = None

    if hasattr(pipe, "vae_scale_factor"):
        factor = pipe.vae_scale_factor
    if hasattr(pipe, "unet"):
        channels = pipe.unet.config.in_channels
        default_size = pipe.unet.config.sample_size
    if hasattr(pipe, "transformer"):
        channels = pipe.transformer.config.in_channels
        default_size = pipe.transformer.config.sample_size
    if hasattr(pipe, "prior_pipe"):
        factor = pipe.prior_pipe.config.resolution_multiple
        channels = pipe.prior_pipe.prior.config.in_channels

    if factor is not None and default_size is None:
        if params["height"].value is None:
            params["height"].value = 1024
        if params["width"].value is None:
            params["width"].value = 1024
        default_size = math.ceil(1024 / factor)

    if factor is not None and channels is not None and default_size is not None:
        size = [
            params["batch_size"].value,
            channels,
            math.ceil(params["height"].value / factor) if params["height"].value else default_size,
            math.ceil(params["width"].value / factor) if params["width"].value else default_size,
        ]
    else:
        print(f'\nModel {params["model"].value} not able to use pre-noised latents.\nNoise options will not be respected.\n')

# INPUT TENSOR }}}

# INPUT ARGS {{{
job = {
    "num_images_per_prompt": params["batch_size"].value,
    "clean_caption": False,  # stop IF nag. what does this even do
}

if not params["negative"].value:
    params["negative"].value = [""]
if params["width"].value:
    job["width"] = params["width"].value
if params["height"].value:
    job["height"] = params["height"].value

if params["iter"].value is Iter.Shuffle:
    jobs = [job] * params["batch_count"].value
    merger = lambda items, name, vals: [i | {name: v} for i, v in zip(items, oversample(vals, len(items)))]  # noqa: E731
else:
    jobs = [job]
    merger = lambda items, name, vals: [i | {name: v} for i in items for v in vals]  # noqa: E731

image_ops = [{}]

for param in params.values():
    if param.multi:
        match param.name:
            case "seed" | "sampler" | "spacing" | "lora":
                pass
            case "pixelate" | "posterize":
                if param.value:
                    image_ops = merger(image_ops, param.name, param.value)
            case other:
                if param.value:
                    jobs = merger(jobs, param.name, param.value)

if schedulers:
    jobs = merger(jobs, "scheduler", schedulers)

i32max = 2**31 - 1
seeds = [torch.randint(high=i32max, low=-i32max, size=(1,)).item()] if not params["seed"].value else params["seed"].value
seeds = [s + c * params["batch_size"].value for c in range(params["batch_count"].value) for s in seeds]
match params["iter"].value:
    case Iter.Shuffle:
        jobs = merger(jobs, "seed", seeds)
    case Iter.Walk:
        jobs = [j | {"seed": s + (n * params["batch_size"].value * params["batch_count"].value)} for s in seeds for n, j in enumerate(jobs)]
    case Iter.Basic:
        jobs = [j | {"seed": s} for s in seeds for j in jobs]
    case other:
        raise ValueError

for j in jobs:
    for key in "prompt", "negative":
        if key in j:
            expands = pexpand(j[key], body=("[", "]"), single=True)
            assert len(expands) == 1
            j[key] = expands[0]

if __name__ != "__main__":  # TODO: this is a hack, pipe shouldn't even be loaded.
    jobs = []

# INPUT ARGS }}}

# INFERENCE {{{
print(f'Generating {len(jobs)} batches of {params["batch_size"].value} images for {len(jobs) * params["batch_size"].value} total...')
filenum = 0
total = len(jobs) * params["batch_size"].value
bar = tqdm.tqdm(desc="Images", total=total) if total > 1 else None
for kwargs in jobs:
    with SmartSigint(job_name="current batch"):
        torch.cuda.empty_cache()
        meta = base_meta.copy()
        seed = kwargs.pop("seed")

        for param in params.values():
            if param.name in kwargs and param.meta:
                meta[param.name] = kwargs[param.name]

        noise_power = kwargs.pop("noise_power") if "noise_power" in kwargs else None
        variance_power = kwargs.pop("variance_power") if "variance_power" in kwargs else None
        variance_scale = kwargs.pop("variance_scale") if "variance_scale" in kwargs else None
        color = kwargs.pop("color") if "color" in kwargs else None
        color_power = kwargs.pop("color_power") if "color_power" in kwargs else None

        if "noise_type" in kwargs:
            noise_type = kwargs.pop("noise_type")
            noise_dtype, noise_device = {
                NoiseType.Cpu16: (torch.float16, "cpu"),
                NoiseType.Cpu16B: (torch.bfloat16, "cpu"),
                NoiseType.Cpu32: (torch.float32, "cpu"),
                NoiseType.Cuda16: (torch.float16, "cuda"),
                NoiseType.Cuda16B: (torch.bfloat16, "cuda"),
                NoiseType.Cuda32: (torch.float32, "cuda"),
            }[noise_type]
        else:
            noise_dtype, noise_device = None, None

        if "scheduler" in kwargs:
            name, sched = kwargs.pop("scheduler")
            pipe.scheduler = sched
            meta["sampler"] = name
            if params["spacing"].value and hasattr(sched.config, "timestep_spacing"):
                meta["spacing"] = sched.config.timestep_spacing

        # NOISE {{{
        generators = [torch.Generator(noise_device).manual_seed(seed + n) for n in range(params["batch_size"].value)]

        if size is not None:
            latents = torch.zeros(size, dtype=dtype, device="cpu")
            for latent, generator in zip(latents, generators):
                # Variance
                if variance_power is not None and variance_scale is not None and variance_power != 0:
                    # save state so init noise seeds are the same with/without
                    state = generator.get_state()
                    variance = torch.randn([1, size[1], variance_scale, variance_scale], generator=generator, dtype=noise_dtype, device=noise_device)
                    latent += torch.nn.UpsamplingBilinear2d((size[2], size[3]))(variance).mul(variance_power)[0].to("cpu")
                    generator.set_state(state)
                # Init noise
                noise = torch.randn(latents.shape[1:], generator=generator, dtype=noise_dtype, device=noise_device)
                if noise_power is not None:
                    noise *= noise_power
                latent += noise.to("cpu")

            # Colored latents
            if XL:
                cols = COLS_XL
            elif SD or isinstance(pipe, PixArtAlphaPipeline):
                cols = COLS_FTMSE
            else:
                cols = None
            if cols and color_power is not None and color is not None and color_power != 0:
                sigma = EulerDiscreteScheduler.from_config(pipe.scheduler.config).init_noise_sigma
                latents += (
                    torch.tensor(cols[color], dtype=dtype, device="cpu")
                    .mul(color_power)
                    .div(sigma)
                    .expand([size[0], size[2], size[3], size[1]])
                    .permute((0, 3, 1, 2))
                )

            kwargs["latents"] = latents

        kwargs["generator"] = generators
        print("seeds:", " ".join([str(seed + n) for n in range(params["batch_size"].value)]))
        # NOISE }}}

        # CONDITIONING {{{
        if compel is not None:
            pos = kwargs.pop("prompt") if "prompt" in kwargs else ""
            neg = kwargs.pop("negative") if "negative" in kwargs else ""
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
        if input_image is not None:
            kwargs["image"] = input_image

        if "prior_num_inference_steps" in pipe_params and "steps" in kwargs:
            steps = kwargs.pop("steps")
            kwargs["prior_num_inference_steps"] = steps
            if "decoder_steps" in kwargs:
                decoder_steps = kwargs.pop("decoder_steps")
                kwargs["num_inference_steps"] = round(math.sqrt(abs(steps * decoder_steps))) if decoder_steps < 0 else decoder_steps

        for f, t in [
            ("steps", ["num_inference_steps"]),
            ("denoise", ["strength"]),
            ("negative", ["negative_prompt"]),
            ("guidance", ["guidance_scale", "prior_guidance_scale"]),
            ("decoder_guidance", ["decoder_guidance_scale"]),
            ("rescale", ["guidance_rescale"]),
        ]:
            if f in kwargs:
                for to in t:
                    kwargs[to] = kwargs[f]
                del kwargs[f]

        # make sure call doesnt err
        for k in list(kwargs.keys()):
            if k not in pipe_params:
                del kwargs[k]

        for n, image in enumerate(pipe(**kwargs).images):
            pnginfo = PngImagePlugin.PngInfo()
            for k, v in meta.items():
                pnginfo.add_text(k, str(v))
            if "latents" in kwargs or n == 0:
                pnginfo.add_text("seed", str(seed + n))
            else:
                pnginfo.add_text("seed", f"{seed} + {n}")

            for ops in image_ops:
                i = image
                info = copy(pnginfo)
                for k, v in ops.items():
                    info.add_text(k, str(v))
                p = params["output"].value.joinpath(f"{filenum:05}.png")
                while p.exists():
                    filenum += 1
                    p = params["output"].value.joinpath(f"{filenum:05}.png")

                if ops.get("posterize", None) is not None:
                    if ops["posterize"] > 1:
                        arr = np.asarray(i).astype(np.float32)
                        factor = float((ops["posterize"] - 1) / 256)
                        i = Image.fromarray(((arr * factor).round() / factor).clip(0, 255).astype(np.uint8), mode=i.mode)

                if ops.get("pixelate", None):
                    if ops["pixelate"] > 1:
                        w, h = i.width, i.height
                        i = i.resize((round(w / ops["pixelate"]), round(h / ops["pixelate"])), resample=Image.BOX)
                        i = i.resize((w, h), resample=Image.NEAREST)

                i.save(p, format="PNG", pnginfo=info, compress_level=4)

        if bar:
            bar.update(params["batch_size"].value)
        # PROCESS }}}
# INFERENCE }}}
