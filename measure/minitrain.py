#! /usr/bin/python
import torch
from tqdm import tqdm
import json
import argparse

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_grad_enabled(False)

def evaluate(channel, latent, tensors, functions):
    l = latent.reshape([21, 21, 21, 4, 1])
    table = l.expand([21, 21, 21, 4, 4]).swapaxes(-1,-2).clone().mul(l)
    for tensor, function in zip(tensors, functions): table = function(table, tensor)
    adjusted = torch.sum(table, axis=[-1,-2])
    diff = (channel - adjusted)
    cumdev = torch.absolute(diff ** 2).sum()
    meandev = diff.mean()
    return cumdev, meandev


def train(channel, latent, tensors, functions, steps, rate):
    dev, mean = evaluate(channel, latent, tensors, functions)

    tensors_best = [tensor.clone() for tensor in tensors]
    dev_best = dev.clone()
    mean_best = mean.clone()

    bar = tqdm(total=steps)
    update = 1000
    for i in range(steps):
        tensors = [tensor + torch.randn([4, 4]) * rate for tensor in tensors_best]
        dev, mean = evaluate(channel, latent, tensors, functions)
        if dev < dev_best: tensors_best, dev_best, mean_best = tensors, dev, mean
        if (i+1) % update == 0:
            bar.update(update)
            bar.set_description(f"{dev_best:.4f} / {mean_best:.4f}")
    return tensors_best, dev_best, mean_best

def sharpen(channel, latent, tensors, functions):
    rate = 10.0
    dev, mean = evaluate(channel, latent, tensors, functions)
    bar = tqdm()
    for r in range(10):
        while True:
            change = False
            for tensor in tensors:
                for x in range(4):
                    for y in range(4):
                        tensor[x,y] += rate
                        ndev, nmean = evaluate(channel, latent, tensors, functions)
                        if ndev < dev:
                            dev, mean = ndev, nmean
                            change = True
                            continue
                        else:
                            tensor[x,y] += rate * -2.0
                        ndev, nmean = evaluate(channel, latent, tensors, functions)
                        if ndev < dev:
                            dev, mean = ndev, nmean
                            change = True
                            continue
                        else: tensor[x,y] += rate
            bar.update(4*4*len(tensors))
            bar.set_description(f"{dev:.4f} / {mean:.4f}")
            if not change: break
        rate /= 10.0
    return tensors, dev, mean



parser = argparse.ArgumentParser(description='Map latent channels to CIELAB')
parser.add_argument('data', type=argparse.FileType(mode='r'))
args = parser.parse_args()

data = json.load(args.data)

assert data['index_space'] == 'SRGB'

latent = torch.tensor(list(map(
    lambda L: list(map(
        lambda A: list(map(
            lambda B: B['latent'],
            A
            )),
        L
        )),
    data['data']
    )))
lab = torch.tensor(list(map(
    lambda L: list(map(
        lambda A: list(map(
            lambda B: B['LAB'],
            A
            )),
        L
        )),
    data['data']
    )))

del data

iterations = [
    (int(1e+5), 1e-1),
    (int(1e+5), 1e-2),
    # (int(1e+6), 1e-3),
]

functions = [torch.mul, torch.add, torch.mul]

for (n, c) in enumerate(["L", "A", "B"]):
    tensors = [torch.randn([4, 4]) for _ in range(len(functions))]

    for (steps, rate) in iterations:
        tensors, dev, mean = train(lab[:,:,:,n], latent, tensors, functions, steps, rate)

    tensors, dev, mean = sharpen(lab[:,:,:,n], latent, tensors, functions)

    print(f"### {c} ###\n{tensors}\n{dev:.8f} / {mean:.8f}\n### {c} ###")
