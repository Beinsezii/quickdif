try:
    import ctypes
    cpixel = ctypes.c_float * 3
    colcon = ctypes.CDLL("./libcolcon.so")

    # up
    colcon.srgb_to_hsv.argtypes = [cpixel]
    colcon.srgb_to_lrgb.argtypes = [cpixel]
    colcon.lrgb_to_xyz.argtypes = [cpixel]
    colcon.xyz_to_lab.argtypes = [cpixel]
    colcon.lab_to_lch.argtypes = [cpixel]

    # down
    colcon.lch_to_lab.argtypes = [cpixel]
    colcon.lab_to_xyz.argtypes = [cpixel]
    colcon.xyz_to_lrgb.argtypes = [cpixel]
    colcon.lrgb_to_srgb.argtypes = [cpixel]
    colcon.srgb_to_hsv.argtypes = [cpixel]

    # extra
    colcon.expand_gamma.argtypes = [ctypes.c_float]
    colcon.expand_gamma.restype = ctypes.c_float
    colcon.correct_gamma.argtypes = [ctypes.c_float]
    colcon.correct_gamma.restype = ctypes.c_float
    colcon.hk_comp_2023.argtypes = [cpixel]

    import json

    has_colcon = True
except Exception:
    has_colcon = False


import argparse

parser = argparse.ArgumentParser(description='Measure cardinal colors on VAE')
parser.add_argument('-v', '--vae', type=str, default='stabilityai/sdxl-vae')
parser.add_argument('-r', '--raw', action='store_true', help='Return raw/unscaled values')
if has_colcon: parser.add_argument('-m', '--map', type=argparse.FileType(mode='w'), help="Output LAB map as .json")
args = parser.parse_args()

import torch, diffusers
from tqdm import tqdm
from diffusers import image_processor

dtype = torch.float32
torch.set_grad_enabled(False)

vae = diffusers.AutoencoderKL.from_pretrained(str(args.vae), use_safetensors=True, torch_dtype=dtype).to('cuda')
processor = image_processor.VaeImageProcessor(2**(len(vae.config.block_out_channels) - 1))

if args.map:
    scale = 5
    iters = (100 // scale) + 1
    data = [[[{} for _ in range(iters)] for _ in range(iters)] for _ in range(iters)]
    bar = tqdm(total=iters**3)
    for L in range(iters):
        for A in range(iters):
            for B in range(iters):
                result = {}
                pix = cpixel(L * scale, A * scale, B * scale)

                lch = cpixel(*pix)
                colcon.lab_to_lch(lch)
                result['LCH'] = list(lch)
                result['LAB'] = list(pix)
                colcon.lab_to_xyz(pix)
                result['XYZ'] = list(pix)
                colcon.xyz_to_lrgb(pix)
                result['LRGB'] = list(pix)
                colcon.lrgb_to_srgb(pix)
                result['SRGB'] = list(pix)

                img = torch.tensor(pix, dtype=dtype).clamp(min=0.0, max=1.0).expand([1, vae.config.sample_size, vae.config.sample_size, 3]).permute((0, 3, 1, 2)).clone()
                img = processor.preprocess(img).to('cuda')
                tensor = vae.encode(img).latent_dist.sample()
                result['latent'] = [c.mean().item() for c in tensor[0]]
                data[L][A][B] = result
                bar.update()
    json.dump({'index_scale': scale, 'index_space': 'LAB', 'vae_scale_factor': vae.config.scaling_factor, 'data': data}, args.map)

else:
    results = {}

    for col, pix in [
        ("black", (0.0, 0.0, 0.0)),
        ("white", (1.0, 1.0, 1.0)),
        ("red", (1.0, 0.0, 0.0)),
        ("green", (0.0, 1.0, 0.0)),
        ("blue", (0.0, 0.0, 1.0)),
        ("cyan", (0.0, 1.0, 1.0)),
        ("magenta", (1.0, 0.0, 1.0)),
        ("yellow", (1.0, 1.0, 0.0)),
    ]:
        img = torch.tensor(pix, dtype=torch.float32).expand([1, vae.config.sample_size, vae.config.sample_size, 3]).permute((0, 3, 1, 2)).clone()
        img = processor.preprocess(img)
        tensor = vae.encode(img.to('cuda')).latent_dist.sample()
        if not args.raw: tensor.mul_(vae.config.scaling_factor)
        results[col] = [c.mean().item() for c in tensor[0]]

    print(results)
