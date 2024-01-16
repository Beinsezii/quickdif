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

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_printoptions(precision=4, linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('data', type=argparse.FileType(mode='r'))
args = parser.parse_args()

data = json.load(args.data)

latent_mean = []
latent_dist = []
rgb = []

for col in data['data']:
    for (k, v) in col.items():
        locals()[k].append(v)

for (k, _) in data['data'][0].items():
    locals()[k] = torch.tensor(locals()[k])

del data

flat = rgb.clone().to('cpu').numpy().flatten()
colcon.convert_space_ffi("srgb".encode(), "oklab".encode(), flat, len(flat))
lab = torch.from_numpy(flat.reshape(rgb.shape)).to(rgb.device)

latent_dist = latent_dist.permute(1,0,2)

network = torch.nn.Sequential(
  torch.nn.Linear(4, 4),
  torch.nn.SiLU(),
  torch.nn.Linear(4, 3),
)

optimizer = torch.optim.AdamW(network.parameters(), lr=1e-5)

latent = latent_dist[5:45 +1]
# color = rgb
color = lab

shape = list(color.shape)
color = color.reshape([1] + shape).expand([latent.shape[0]] + shape)
color_scale = 100

iter = tqdm(range(int(1e+6)))
deviations = torch.tensor([0.5, 0.75, 0.9, 0.99, 1.0])

try:
    for epoch in iter:
        optimizer.zero_grad()
        prediction = network(latent)
        loss = (prediction - color).absolute()
        devs = loss.clone().quantile(deviations).mul(color_scale).tolist()
        loss = (loss).sum()
        display_loss = loss.item()
        loss.backward()
        optimizer.step()

        devs = ' '.join(map(lambda d: f"{d:.1f}", devs))
        iter.desc = f"Loss {loss:.1f} Deviations {devs}"

except KeyboardInterrupt:
    print()

print("###\n")
for p in network.parameters():
    print(f"{p.t()}\n")
print("###")
