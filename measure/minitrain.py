#! /usr/bin/env python

### COLCON ### {{{
import numpy
import ctypes, os
from sys import platform

cpixel = ctypes.c_float * 3
# cpixels = ctypes.POINTER(ctypes.c_float)
cpixels = numpy.ctypeslib.ndpointer(ndim=1, flags=('W', 'C', 'A'))

if platform == "win32":
    LIBRARY = "colcon.dll"
elif platform == "darwin":
    LIBRARY = "libcolcon.dylib"
elif platform == "linux":
    LIBRARY = "libcolcon.so"

colcon = ctypes.CDLL(os.path.join(os.path.dirname(os.path.realpath(__file__)), LIBRARY))

colcon.convert_space_ffi.argtypes = [ctypes.c_char_p, ctypes.c_char_p, cpixels, ctypes.c_uint]
colcon.convert_space_ffi.restype = ctypes.c_int32

# up
colcon.srgb_to_hsv.argtypes = [cpixel]
colcon.srgb_to_lrgb.argtypes = [cpixel]
colcon.lrgb_to_xyz.argtypes = [cpixel]
colcon.xyz_to_lab.argtypes = [cpixel]
colcon.xyz_to_oklab.argtypes = [cpixel]
colcon.xyz_to_jzazbz.argtypes = [cpixel]
colcon.lab_to_lch.argtypes = [cpixel]

# down
colcon.lch_to_lab.argtypes = [cpixel]
colcon.jzazbz_to_xyz.argtypes = [cpixel]
colcon.oklab_to_xyz.argtypes = [cpixel]
colcon.lab_to_xyz.argtypes = [cpixel]
colcon.xyz_to_lrgb.argtypes = [cpixel]
colcon.lrgb_to_srgb.argtypes = [cpixel]
colcon.srgb_to_hsv.argtypes = [cpixel]

# extra
colcon.srgb_eotf.argtypes = [ctypes.c_float]
colcon.srgb_eotf.restype = ctypes.c_float
colcon.srgb_eotf_inverse.argtypes = [ctypes.c_float]
colcon.srgb_eotf_inverse.restype = ctypes.c_float
colcon.hk_high2023.argtypes = [cpixel]
colcon.hk_high2023_comp.argtypes = [cpixel]
### COLCON ### }}}

import torch
import json
import argparse
from tqdm import tqdm
import safetensors.torch
from PIL import Image
from pathlib import Path
import numpy as np

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_printoptions(precision=4, linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('data', type=argparse.FileType(mode='r'))
parser.add_argument('-l', '--latent', type=Path)
parser.add_argument('-p', '--preview', type=Path, default='/tmp/minitrain_preview.png')
args = parser.parse_args()

data = json.load(args.data)

if args.latent and args.preview:
    latent_preview = safetensors.torch.load_file(args.latent)
    latent_preview = latent_preview['latent_tensor'][0].permute(1,2,0)
    preview_shape = list(latent_preview.shape)
    preview_shape[-1] = 3

latent_mean = []
latent_dist = []
rgb = []
vae_factor = data['vae_factor']

for col in data['data']:
    for (k, v) in col.items():
        locals()[k].append(v)

for (k, _) in data['data'][0].items():
    locals()[k] = torch.tensor(locals()[k])

del data


latent_dist = latent_dist.permute(1,0,2)

network = torch.nn.Sequential(
  torch.nn.Linear(4, 4),
  torch.nn.SiLU(),
  torch.nn.Linear(4, 4),
  torch.nn.SiLU(),
  torch.nn.Linear(4, 3),
)

color_format = "oklab"
color_scale = 100

optimizer = torch.optim.AdamW(network.parameters(), lr=3e-5)

# latent = latent_dist[5:45 +1]
latent = latent_dist

flat = rgb.clone().to('cpu').numpy().flatten()
colcon.convert_space_ffi("srgb".encode(), color_format.encode(), flat, len(flat))
color = torch.from_numpy(flat.reshape(rgb.shape)).to(rgb.device)

shape = list(color.shape)
color = color.reshape([1] + shape).expand([latent.shape[0]] + shape)

iter = tqdm(range(int(1e+6)))
deviations = torch.tensor([0.5, 0.75, 0.9, 0.99, 1.0])

preview = 1000

outlier_weight = 0.01
slope = 3
loss_weight = (1.0 - ((torch.linspace(0.0,1.0,color.shape[0]) - 0.5) / (0.5 / (1.0 - outlier_weight ** slope))).abs()) ** (1/slope)
loss_weight = loss_weight.reshape([color.shape[0], 1, 1]).expand(color.shape)

try:
    for step in iter:
        optimizer.zero_grad()
        prediction = network(latent)
        loss = (prediction - color).absolute() * color_scale
        devs = loss.clone().quantile(deviations)
        loss = (loss * loss_weight).mean()
        display_loss = loss.item()
        loss.backward()
        optimizer.step()

        devs = ' '.join(map(lambda d: f"{d:.3f}", devs))
        iter.desc = f"Loss {loss.item():.4f} Deviations {devs}"

        if (step + 1) % preview == 0 and latent_preview is not None:
            with torch.no_grad():
                flat = network(latent_preview).cpu().numpy().flatten()
                colcon.convert_space_ffi(color_format.encode(), "srgb".encode(), flat, len(flat))
                Image.fromarray(np.uint8(flat.reshape(preview_shape).clip(0.0,1.0) * 255), mode="RGB").save(args.preview, compress_level=0)

except KeyboardInterrupt:
    print()

print("###\n")
for p in network.parameters():
    print(f"{p.t()}\n")
print("###")
