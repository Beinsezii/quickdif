import argparse
import enum
import functools
import json
import random
import re
import signal
from copy import copy
from inspect import signature
from io import BytesIO, UnsupportedOperation
from math import sqrt
from pathlib import Path
from sys import exit
from typing import Any

import numpy as np
import numpy.linalg as npl
import tomllib
from PIL import Image, PngImagePlugin
from tqdm import tqdm

# MATRICES {{{
COLS_XL = {
    # {{{
    "black": [-2.8232, 0.5033, 0.3139, 0.3359],
    "white": [2.3501, 0.2248, 1.2127, -1.0597],
    "red": [-2.5614, -2.5784, 1.3915, -1.6186],
    "green": [-0.4599, 1.8333, 3.4502, 1.1301],
    "blue": [0.0593, 2.1290, -2.3017, 0.5399],
    "cyan": [1.6195, 3.3881, 0.5599, 1.0360],
    "magenta": [-0.1252, -0.6654, -1.5711, -1.1750],
    "yellow": [-0.8608, -1.3759, 4.2304, -1.0693],
}  # }}}
COLS_FTMSE = {
    # {{{
    "black": [-0.9953, -2.6024, 1.1153, 1.2966],
    "white": [2.1749, 1.4434, -0.0318, -1.1621],
    "red": [1.2575, -0.8768, -1.8788, 0.7792],
    "green": [0.7681, 0.8218, 2.3520, 1.8135],
    "blue": [-0.6330, -3.0466, 0.9888, -1.4022],
    "cyan": [0.5828, -0.1041, 2.9347, -0.0978],
    "magenta": [0.7387, -1.5557, -1.3593, -1.4386],
    "yellow": [2.3476, 2.0031, -0.0791, 1.6609],
}  # }}}

XYZ_M1 = np.array(
    # {{{
    [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]
).T  # }}}

OKLAB_M1 = np.array(
    # {{{
    [
        [0.8189330101, 0.0329845436, 0.0482003018],
        [0.3618667424, 0.9293118715, 0.2643662691],
        [-0.1288597137, 0.0361456387, 0.6338517070],
    ]
)  # }}}
OKLAB_M2 = np.array(
    # {{{
    [
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ]
)  # }}}

# }}}


# UTILS {{{
def oversample(population: list, k: int):
    samples = []
    while len(samples) < k:
        samples += random.sample(population, min(len(population), k - len(samples)))
    assert len(samples) == k
    return samples


def roundint(n: int | float, step: int) -> int:
    if n % step >= step / 2:
        return round(n + step - (n % step))
    else:
        return round(n - (n % step))


def spowf(array: np.ndarray, pow: int | float | list[int | float]) -> np.ndarray:
    return np.copysign(abs(array) ** pow, array)


def lrgb_to_oklab(array: np.ndarray) -> np.ndarray:
    return (spowf((array) @ (XYZ_M1 @ OKLAB_M1), 1 / 3)) @ OKLAB_M2


def oklab_to_lrgb(array: np.ndarray) -> np.ndarray:
    return spowf((array @ npl.inv(OKLAB_M2)), 3) @ (npl.inv(OKLAB_M1) @ npl.inv(XYZ_M1))


# }}}


# pexpand {{{
@functools.cache
def _pexpand_bounds(string: str, body: tuple[str, str]) -> None | tuple[int, int]:
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


def _pexpand(prompt: str, body: tuple[str, str] = ("{", "}"), sep: str = "|", single: bool = False) -> list[str]:
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


def pexpand(prompt: str, body: tuple[str, str] = ("{", "}"), sep: str = "|", single: bool = False) -> list[str]:
    if single:
        return _pexpand(prompt, body, sep, single)
    else:
        return _pexpand_cache(prompt, body, sep, single)


# }}}


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
    Unipc = enum.auto()
    UnipcK = enum.auto()
    Unipc2 = enum.auto()
    Unipc2K = enum.auto()
    Unipc3 = enum.auto()
    Unipc3K = enum.auto()


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


# }}}


class Resolution:
    # {{{
    def __init__(self, resolution: str | tuple[int, int]):
        if isinstance(resolution, str):
            self._str = resolution
            m = re.match(r"^ *([\d\.]+) *: *([\d\.]+) *(?:: *(\d+))? *(?:([@^]) *([\d\.]+))? *$", resolution)
            if m:
                hor, ver, rnd, method, mpx = m.groups()
                hor, ver = float(hor), float(ver)
                rnd = 64 if rnd is None else int(rnd)
                mpx = 1.0 if mpx is None else float(mpx)
                if method == "^":
                    mpx = mpx * mpx / 10**6
                self._width = roundint(sqrt(hor / ver * mpx * 10**6), rnd)
                self._height = roundint(sqrt(ver / hor * mpx * 10**6), rnd)
            else:
                m = re.match(r"^ *(\d+) *[x*]? *(\d+)? *$", resolution)
                if m is None:
                    m = re.match(r"^ *(\d+)? *[x*] *(\d+) *$", resolution)
                if m:
                    w, h = m.groups()
                    w = 1024 if w is None else int(w)
                    h = 1024 if h is None else int(h)
                    self._width, self._height = w, h
                else:
                    raise ValueError
        else:
            self._str = None
            self._width, self._height = resolution
        if not (16 <= self.width <= 4096 and 16 <= self.height <= 4096):
            raise ValueError

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __repr__(self):
        return str(self.width) + "x" + str(self.height)

    def __str__(self):
        return self._str if self._str is not None else self.__repr__()

    # }}}


class QDParam:
    # {{{
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

    def _cast(self, new):
        return new if isinstance(new, self.typing) else self.typing(new)

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
                new = [self._cast(v) for v in new]
                self._value = new
            else:
                raise ValueError(f"Refusing to assign list '{new}' to non-multi QDParam '{self.name}'")
        else:
            if self.multi:
                self._value = [self._cast(new)]
            else:
                self._value = self._cast(new)

    # }}}


def qdparams() -> dict[str, QDParam]:
    # {{{
    params = [
        ### Batching
        QDParam("prompt", str, multi=True, meta=True, help="Positive prompt"),
        QDParam("negative", str, short="-n", long="--negative", multi=True, meta=True, help="Negative prompt"),
        QDParam("seed", int, short="-e", long="--seed", multi=True, meta=True, help="Seed for RNG"),
        QDParam(
            "resolution",
            Resolution,
            short="-r",
            long="--resolution",
            multi=True,
            help="Resolution in either [width]x[height] or aspect_x:aspect_y[:round][@megapixels|^square] formats.",
        ),
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
            "decoder_guidance",
            float,
            short="-dg",
            long="--decoder-guidance",
            multi=True,
            meta=True,
            help="Guidance for the Decoder stage if applicable",
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
        QDParam(
            "power",
            float,
            long="--power",
            multi=True,
            meta=True,
            help="Simple filter which scales final image values away from gray based on an exponent",
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
euler/ddim/etc. - Literal names;
k - Use karras sigmas;
s - Use SDE stochastic noise;
a - Use ancestral sampling;
2/3 - Use 2nd/3rd order sampling;
Ex. 'sdpm2k' is equivalent to 'DPM++ 2M SDE Karras'""",
        ),
        QDParam("spacing", Spacing, long="--spacing", value=Spacing.Trailing, multi=True, meta=True, help="Sampler timestep spacing"),
        ### Global
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
    ]  # ugh autoindent
    return {param.name: param for param in params}
    # }}}


def build_parser(params: dict[str, QDParam]) -> argparse.ArgumentParser:
    # {{{
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
    # parser.add_argument("--toml", type=argparse.FileType(mode="a+b"), help="Output settings to TOML")
    parser.add_argument("--comment", type=str, help="Add a comment to the image.")
    parser.add_argument("--print", action="store_true", help="Print out generation params and exit.")
    parser.add_argument("--help", action="help")

    return parser
    # }}}


def parse_cli(params: dict[str, QDParam]) -> tuple[str | None, Image.Image | None]:
    # {{{
    args = vars(build_parser(params).parse_args())

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
                    params["resolution"].value = (meta_image.width, meta_image.height)
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
        if (
            id in params
            and val is not None
            and not (isinstance(val, list) and len(val) == 0 and params[id].long is None and params[id].short is None)
        ):
            params[id].value = val

    args = {k: v for k, v in args.items() if k not in params}

    if args.get("json", None) is not None:
        dump = {}
        for k, v in params.items():
            if v.value != v.default:
                v = v.value
                if isinstance(v, Path):
                    v = str(v)
                elif isinstance(v, list):
                    if all(map(lambda x: isinstance(x, Resolution), v)):
                        v = [str(v) for v in v]
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

    comment = None
    if args.get("comment", ""):
        try:
            with open(args["comment"], "r") as f:
                comment = f.read()
        except Exception:
            comment = args["comment"]

    return (comment, input_image)
    # }}}


disable_amd_patch = True
if __name__ == "__main__":
    cli_params = qdparams()
    (cli_comment, cli_image) = parse_cli(cli_params)
    disable_amd_patch = cli_params["disable_amd_patch"].value

# TORCH PRELUDE {{{
#
# Load Torch and libs that depend on it after the CLI cause it's laggy.
import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

amd_hijack = False
if not disable_amd_patch:
    if "AMD" in torch.cuda.get_device_name() or "Radeon" in torch.cuda.get_device_name():
        try:
            from flash_attn import flash_attn_func

            sdpa = torch.nn.functional.scaled_dot_product_attention

            def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                if query.shape[3] <= 128 and attn_mask is None and query.dtype != torch.float32:
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

from compel import Compel, ReturnedEmbeddingsType  # noqa: E402
from transformers import CLIPTokenizer  # noqa: E402

from diffusers import (  # noqa: E402
    AutoencoderKL,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PixArtAlphaPipeline,
    SchedulerMixin,
    StableCascadeCombinedPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
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
# }}}


# elegent solution from <https://stackoverflow.com/questions/842557/>
class SmartSigint:
    # {{{
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

    # }}}


def get_pipe(model: str, offload: Offload, dtype: DType, img2img: bool) -> DiffusionPipeline:
    # {{{
    pipe_args = {
        "add_watermarker": False,
        "safety_checker": None,
        "torch_dtype": {DType.FP16: torch.float16, DType.BF16: torch.bfloat16, DType.FP32: torch.float32}[dtype],
        "use_safetensors": True,
        "watermarker": None,
    }

    if "stabilityai/stable-cascade" in model:
        if pipe_args["torch_dtype"] == torch.float16:
            pipe_args["torch_dtype"] = torch.bfloat16
        pipe = StableCascadeCombinedPipeline.from_pretrained(model, **pipe_args)
    elif model.endswith(".safetensors"):
        if img2img:
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(model, **pipe_args)
            except:  # noqa: E722
                pipe = StableDiffusionImg2ImgPipeline.from_single_file(model, **pipe_args)
        else:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(model, **pipe_args)
            except:  # noqa: E722
                pipe = StableDiffusionPipeline.from_single_file(model, **pipe_args)
    else:
        if img2img:
            pipe = AutoPipelineForImage2Image.from_pretrained(model, **pipe_args)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model, **pipe_args)

    pipe.safety_checker = None
    pipe.watermarker = None
    match offload:
        case Offload.NONE:
            if hasattr(pipe, "prior_pipe"):
                pipe.prior_pipe = pipe.prior_pipe.to("cuda")
            if hasattr(pipe, "decoder_pipe"):
                pipe.decoder_pipe = pipe.decoder_pipe.to("cuda")
            pipe = pipe.to("cuda")
        case Offload.Model:
            pipe.enable_model_cpu_offload()
        case Offload.Sequential:
            pipe.enable_sequential_cpu_offload()
        case _:
            raise ValueError

    return pipe
    # }}}


def is_xl(pipe: DiffusionPipeline) -> bool:
    return isinstance(pipe, StableDiffusionXLPipeline) or isinstance(pipe, StableDiffusionXLImg2ImgPipeline)


def is_sd(pipe: DiffusionPipeline) -> bool:
    return isinstance(pipe, StableDiffusionPipeline) or isinstance(pipe, StableDiffusionImg2ImgPipeline)


def set_attn(pipe: DiffusionPipeline, attention: Attention):
    # {{{
    processor = None
    match attention:
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

    for item in [pipe] + [
        getattr(pipe, id, None)
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


def apply_loras(loras: list[str], pipe: DiffusionPipeline) -> str | None:
    # {{{
    adapters = []
    for n, lora in enumerate(loras):
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
        return "\x1f".join(map((lambda a: a["path"] if a["scale"] == 1.0 else f'{a["path"]}:::{a["scale"]}'), adapters))
    else:
        return None
    # }}}


def get_compel(pipe: DiffusionPipeline) -> Compel | None:
    # {{{
    if hasattr(pipe, "tokenizer") and isinstance(pipe.tokenizer, CLIPTokenizer):
        if hasattr(pipe, "tokenizer_2"):
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

    return compel
    # }}}


def set_vae(params: dict[str, QDParam], pipe: DiffusionPipeline):
    # {{{
    if is_xl(pipe) and params["xl_vae"].value:
        pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=pipe.vae.dtype, use_safetensors=True).to(pipe.vae.device)
    if params["tile"].value:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.enable_slicing()
    if pipe.dtype != torch.float16:
        pipe.vae.config.force_upcast = False
    # }}}


def build_schedulers(params: dict[str, QDParam], default_scheduler: Any) -> list[tuple[dict[str, str], type[SchedulerMixin]]]:
    # {{{
    sampler_args = {
        "steps_offset": 0,
        "set_alpha_to_one": True,
        "final_sigmas_type": "zero",
    }
    sampler_map: dict[Sampler, tuple[Any, dict[str, Any]]] = {
        Sampler.Default: (default_scheduler, {}),
        Sampler.Ddim: (DDIMScheduler, {}),
        Sampler.Ddpm: (DDPMScheduler, {}),
        Sampler.Euler: (EulerDiscreteScheduler, {}),
        Sampler.EulerK: (EulerDiscreteScheduler, {"use_karras_sigmas": True}),
        Sampler.EulerA: (EulerAncestralDiscreteScheduler, {}),
        Sampler.Dpm: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 1, "use_karras_sigmas": False}),
        Sampler.DpmK: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 1, "use_karras_sigmas": True}),
        Sampler.SDpm: (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "solver_order": 1, "use_karras_sigmas": False}),
        Sampler.SDpmK: (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "solver_order": 1, "use_karras_sigmas": True}),
        Sampler.Dpm2: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": False}),
        Sampler.Dpm2K: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
        Sampler.SDpm2: (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "solver_order": 2, "use_karras_sigmas": False}),
        Sampler.SDpm2K: (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
        Sampler.Dpm3: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 3, "use_karras_sigmas": False}),
        Sampler.Dpm3K: (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 3, "use_karras_sigmas": True}),
        Sampler.Unipc: (UniPCMultistepScheduler, {"solver_order": 1, "use_karras_sigmas": False}),
        Sampler.UnipcK: (UniPCMultistepScheduler, {"solver_order": 1, "use_karras_sigmas": True}),
        Sampler.Unipc2: (UniPCMultistepScheduler, {"solver_order": 2, "use_karras_sigmas": False}),
        Sampler.Unipc2K: (UniPCMultistepScheduler, {"solver_order": 2, "use_karras_sigmas": True}),
        Sampler.Unipc3: (UniPCMultistepScheduler, {"solver_order": 3, "use_karras_sigmas": False}),
        Sampler.Unipc3K: (UniPCMultistepScheduler, {"solver_order": 3, "use_karras_sigmas": True}),
    }
    for name, (sched, args) in sampler_map.items():
        for k in args:
            sig = signature(sched).parameters
            if k not in sig:
                raise AssertionError(f"kwarg '{k}' not valid for requested scheduler for '{name}'.\nParameters: {list(sig)}")

    builders: list[tuple[dict[str, str], Any, dict[str, Any]]] = []
    if not params["sampler"].value:
        params["sampler"].value = [Sampler.Default]
    if params["sampler"].value:
        for s in params["sampler"].value:
            sched, kwargs = sampler_map[s]
            builders.append(({"sampler": s}, sched, kwargs))

    if params["spacing"].value:
        builders = [
            (sched_meta | {"spacing": space}, sched, kwargs | {"timestep_spacing": space})
            for sched_meta, sched, kwargs in builders
            for space in params["spacing"].value
        ]

    for sched_meta, _, _ in builders:
        if sched_meta["sampler"] == Sampler.Default:
            del sched_meta["sampler"]

    return [(sched_meta, sched.from_config(default_scheduler.config, **(sampler_args | kwargs))) for sched_meta, sched, kwargs in builders]
    # }}}


def get_latent_params(pipe: DiffusionPipeline) -> tuple[int, float, int] | None:
    # {{{
    factor: float | None = None
    channels: int | None = None
    default_size: int | None = None

    if hasattr(pipe, "vae_scale_factor"):
        factor = pipe.vae_scale_factor
    if hasattr(pipe, "unet"):
        channels = pipe.unet.config.get("in_channels", None)
        default_size = pipe.unet.config.get("sample_size", None)
    if hasattr(pipe, "transformer"):
        channels = pipe.transformer.config.get("in_channels", None)
        default_size = pipe.transformer.config.get("sample_size", None)
    if hasattr(pipe, "prior_pipe"):
        factor = pipe.prior_pipe.config.get("resolution_multiple", None)
        channels = pipe.prior_pipe.prior.config.get("in_channels", None)

    if factor is not None and channels is not None:
        return (channels, factor, default_size if default_size is not None else round(1024 / factor))
    else:
        return None
    # }}}


def build_jobs(params: dict[str, QDParam], schedulers: list[tuple[dict[str, str], type[SchedulerMixin]]]) -> list[dict[str, Any]]:
    # {{{
    job = {
        "num_images_per_prompt": params["batch_size"].value,
        "clean_caption": False,  # stop IF nag. what does this even do
    }

    if not params["negative"].value:
        params["negative"].value = [""]

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
                case "power" | "pixelate" | "posterize":
                    if param.value:
                        image_ops = [i | {param.name: v} for i in image_ops for v in param.value]
                case _:
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
        case _:
            raise ValueError

    if params["iter"].value is Iter.Shuffle:
        jobs = [j | {"image_ops": [o]} for j, o in zip(jobs, oversample(image_ops, len(jobs)))]
    else:
        # Shouldn't need to copy since image_ops isn't mutated
        jobs = [j | {"image_ops": image_ops} for j in jobs]

    for j in jobs:
        for key in "prompt", "negative":
            if key in j:
                expands = pexpand(j[key], body=("[", "]"), single=True)
                assert len(expands) == 1
                j[key] = expands[0]

    return jobs
    # }}}


def process_job(
    params: dict[str, QDParam],
    pipe: DiffusionPipeline,
    job: dict[str, Any],
    meta: dict[str, str],
    input_image: Image.Image | None,
):
    # {{{
    torch.cuda.empty_cache()
    seed = job.pop("seed")

    for param in params.values():
        if param.name in job and param.meta:
            meta[param.name] = job[param.name]

    resolution: Resolution | None = job.pop("resolution") if "resolution" in job else None
    noise_power = job.pop("noise_power") if "noise_power" in job else None
    variance_power = job.pop("variance_power") if "variance_power" in job else None
    variance_scale = job.pop("variance_scale") if "variance_scale" in job else None
    color = job.pop("color") if "color" in job else None
    color_power = job.pop("color_power") if "color_power" in job else None
    image_ops: list[dict[str, Any]] = job.pop("image_ops")

    if "noise_type" in job:
        noise_type = job.pop("noise_type")
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

    if "scheduler" in job:
        sched_meta, sched = job.pop("scheduler")
        meta |= sched_meta
        pipe.scheduler = sched

    # NOISE {{{
    generators = [torch.Generator(noise_device).manual_seed(seed + n) for n in range(params["batch_size"].value)]
    latent_params = get_latent_params(pipe)

    if input_image is None and latent_params is not None:
        channels, factor, default_size = latent_params
        if resolution is not None:
            width, height = round(resolution.width / factor), round(resolution.height / factor)
        else:
            width, height = default_size, default_size
            job["width"], job["height"] = round(default_size * factor), round(default_size * factor)
        shape = (params["batch_size"].value, channels, height, width)
        latents = torch.zeros(shape, dtype=pipe.dtype, device="cpu")
        for latent, generator in zip(latents, generators):
            # Variance
            if variance_power is not None and variance_scale is not None and variance_power != 0:
                # save state so init noise seeds are the same with/without
                state = generator.get_state()
                variance = torch.randn([1, shape[1], variance_scale, variance_scale], generator=generator, dtype=noise_dtype, device=noise_device)
                latent += torch.nn.UpsamplingBilinear2d((shape[2], shape[3]))(variance).mul(variance_power)[0].to("cpu")
                generator.set_state(state)
            # Init noise
            noise = torch.randn(latents.shape[1:], generator=generator, dtype=noise_dtype, device=noise_device)
            if noise_power is not None:
                noise *= noise_power
            latent += noise.to("cpu")

        # Colored latents
        if is_xl(pipe):
            cols = COLS_XL
        elif is_sd(pipe) or isinstance(pipe, PixArtAlphaPipeline):
            cols = COLS_FTMSE
        else:
            cols = None
        if cols and color_power is not None and color is not None and color_power != 0:
            sigma = EulerDiscreteScheduler.from_config(pipe.scheduler.config).init_noise_sigma
            latents += (
                torch.tensor(cols[color], dtype=pipe.dtype, device="cpu")
                .mul(color_power)
                .div(sigma)
                .expand([shape[0], shape[2], shape[3], shape[1]])
                .permute((0, 3, 1, 2))
            )

        job["latents"] = latents

    job["generator"] = generators
    print("seeds:", " ".join([str(seed + n) for n in range(params["batch_size"].value)]))
    # NOISE }}}

    # CONDITIONING {{{
    compel = get_compel(pipe)
    if compel is not None:
        pos = job.pop("prompt") if "prompt" in job else ""
        neg = job.pop("negative") if "negative" in job else ""
        if is_xl(pipe):
            ncond, npool = compel.build_conditioning_tensor(neg)
            pcond, ppool = compel.build_conditioning_tensor(pos)
            job = job | {"pooled_prompt_embeds": ppool, "negative_pooled_prompt_embeds": npool}
        else:
            pcond = compel.build_conditioning_tensor(pos)
            ncond = compel.build_conditioning_tensor(neg)
        pcond, ncond = compel.pad_conditioning_tensors_to_same_length([pcond, ncond])
        job |= {"prompt_embeds": pcond, "negative_prompt_embeds": ncond}
    # CONDITIONING }}}

    # INFERENCE {{{
    pipe_params = signature(pipe).parameters

    if input_image is not None:
        if resolution is not None:
            job["image"] = input_image.resize(resolution.resolution, Image.LANCZOS)
        else:
            job["image"] = input_image
    elif resolution is not None:
        job["width"], job["height"] = resolution.resolution

    if "prior_num_inference_steps" in pipe_params and "steps" in job:
        steps = job.pop("steps")
        job["prior_num_inference_steps"] = steps
        if "decoder_steps" in job:
            decoder_steps = job.pop("decoder_steps")
            job["num_inference_steps"] = round(sqrt(abs(steps * decoder_steps))) if decoder_steps < 0 else decoder_steps

    for f, t in [
        ("steps", ["num_inference_steps"]),
        ("denoise", ["strength"]),
        ("negative", ["negative_prompt"]),
        ("guidance", ["guidance_scale", "prior_guidance_scale"]),
        ("decoder_guidance", ["decoder_guidance_scale"]),
        ("rescale", ["guidance_rescale"]),
    ]:
        if f in job:
            for to in t:
                job[to] = job[f]
            del job[f]

    # make sure call doesnt err
    for k in list(job.keys()):
        if k not in pipe_params:
            del job[k]

    filenum = 0
    for n, image_array in enumerate(pipe(output_type="np", **job).images):
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        if "latents" in job or n == 0:
            pnginfo.add_text("seed", str(seed + n))
        else:
            pnginfo.add_text("seed", f"{seed} + {n}")

        for ops in image_ops:
            info = copy(pnginfo)
            for k, v in ops.items():
                info.add_text(k, str(v))
            p = params["output"].value.joinpath(f"{filenum:05}.png")
            while p.exists():
                filenum += 1
                p = params["output"].value.joinpath(f"{filenum:05}.png")

            # Direct array ops
            op_arr: np.ndarray = np.asarray(image_array)  # mutable reference to make pyright happy

            if "power" in ops:
                # ^2.2 for approx sRGB EOTF
                okl = lrgb_to_oklab(spowf(op_arr, 2.2))

                # ≈1/3 cause OKL's top heavy lightness curve
                offset = [0.35, 1, 1]
                # Halve chromacities' power slope
                power = [ops["power"], sqrt(ops["power"]), sqrt(ops["power"])]

                okl = spowf((okl + offset), power) - offset

                # back to sRGB with approx OETF
                op_arr = spowf(oklab_to_lrgb(okl), 1 / 2.2)

            if "posterize" in ops:
                if ops["posterize"] > 1:
                    factor = float((ops["posterize"] - 1) / 256)
                    op_arr = (op_arr * 255 * factor).round() / factor / 255

            # PIL ops
            op_pil: Image.Image = Image.fromarray((op_arr * 255).clip(0, 255).astype(np.uint8))
            del op_arr

            if "pixelate" in ops:
                if ops["pixelate"] > 1:
                    w, h = op_pil.width, op_pil.height
                    op_pil = op_pil.resize((round(w / ops["pixelate"]), round(h / ops["pixelate"])), resample=Image.BOX)
                    op_pil = op_pil.resize((w, h), resample=Image.NEAREST)

            op_pil.save(p, format="PNG", pnginfo=info, compress_level=4)
    # }}}
    # }}}


def main(params: dict[str, QDParam], meta: dict[str, str], image: Image.Image | None):
    # {{{
    with SmartSigint(num=2, job_name="model load"):
        pipe = get_pipe(params["model"].value, params["offload"].value, params["dtype"].value, image is not None)

    if not get_latent_params(pipe):
        print(f'\nModel {params["model"].value} not able to use pre-noised latents.\nNoise options will not be respected.\n')

    if params["lora"].value:
        lora_string = apply_loras(params["lora"].value, pipe)
        if lora_string:
            meta["lora"] = lora_string

    if hasattr(pipe, "vae"):
        set_vae(params, pipe)

    if params["compile"].value:
        if hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet)
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer)
        if hasattr(pipe, "prior_pipe"):
            pipe.prior_pipe.prior = torch.compile(pipe.prior_pipe.prior)
        if hasattr(pipe, "decoder_pipe"):
            pipe.decoder_pipe.decoder = torch.compile(pipe.decoder_pipe.decoder)
    elif params["attention"].value:
        set_attn(pipe, params["attention"].value)

    if hasattr(pipe, "scheduler"):
        schedulers = build_schedulers(params, pipe.scheduler)
        if len(schedulers) == 1:
            sched_meta, sched = schedulers[0]
            meta |= sched_meta
            pipe.scheduler = sched
    else:
        schedulers = []

    jobs = build_jobs(params, schedulers)

    total_images = len(jobs) * params["batch_size"].value
    print(f'\nGenerating {len(jobs)} batches of {params["batch_size"].value} images for {total_images} total...')
    pbar = tqdm(desc="Images", total=total_images, smoothing=0)
    for job in jobs:
        with SmartSigint(job_name="current batch"):
            process_job(params, pipe, job, meta.copy(), image)
            pbar.update(params["batch_size"].value)

    # }}}


if __name__ == "__main__":
    params = locals()["cli_params"]
    comment = locals()["cli_comment"]
    image = locals()["cli_image"]
    meta: dict[str, str] = {"model": params["model"].value, "url": "https://github.com/Beinsezii/quickdif"}
    if comment:
        meta["comment"] = comment

    main(params, meta, image)
