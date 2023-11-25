#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

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

parser = argparse.ArgumentParser(description='Map latent channels to CIELAB')
parser.add_argument('data', type=argparse.FileType(mode='r'))
args = parser.parse_args()

data = json.load(args.data)

latent = np.array(list(map(
    lambda L: list(map(
        lambda A: list(map(
            lambda B: B['latent'],
            A
            )),
        L
        )),
    data['data']
    )))

lab = np.array(list(map(
    lambda L: list(map(
        lambda A: list(map(
            lambda B: B['LAB'],
            A
            )),
        L
        )),
    data['data']
    )))

## CIE L
fig, axes = plt.subplots(3, 3)
fig.suptitle('Latent to CIE L* A 0,50,100 B 0,50,100')
for row, a in zip(axes, [0, 10, 20]):
    for ax, b in zip(row, [0, 10, 20]):
        ax.set(xlim=(0, 100), xticks=np.arange(0, 100, 25),
               ylim=(-25, 125), yticks=np.arange(-25, 125, 25),
               aspect=1
           )
        c1, c2, c3, c4 = np.moveaxis(latent, 0, -1)[a][b]
        chroma = np.sqrt(c2 ** 2 + c3 ** 2)
        hue = np.degrees(np.arctan2(c3, c2))
        l = np.moveaxis(lab, 0, -1)[a][b][0]
        delta32 = c3 - c2

        polys = np.moveaxis(np.array([
            np.polyfit(l, c1, 1),
            np.polyfit(l, c2, 1),
            np.polyfit(l, c3, 1),
            np.polyfit(l, c4, 1),
        ]), 0, -1)
        fit1234 = l * polys[0].mean() + polys[1].mean()

        # Raw channels
        ax.plot(l, c1, 'r-')
        ax.plot(l, c2, 'g-')
        ax.plot(l, c3, 'b-')
        ax.plot(l, c4, 'y-')
        ax.plot(l, chroma, 'm--')
        ax.plot(l, delta32, 'm+')
        ax.plot(l, fit1234, 'k--')
        ax.plot(l, l, 'k')

        # Corrected C1
        c1 -= -21.675973892211914
        c1 *= 100 / (abs(-21.675973892211914) + 18.038631439208984)
        ax.plot(l, c1, 'r+')

        # Corrected C4
        c4 -= 2.5792038440704346
        c4 *= -100 / (abs(-15.211228370666504) + 2.5792038440704346)
        ax.plot(l, c4, 'y+')

        # Naive joined corrections
        ax.plot(l, (c1 + c4) / 2, 'c.')

plt.show()
