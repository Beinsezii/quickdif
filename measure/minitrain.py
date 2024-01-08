#! /usr/bin/python
import torch
from tqdm import tqdm
import json
import argparse

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_grad_enabled(False)

def evaluate(channel, latent, tensors, functions):
    l = latent.clone()
    for tensor, function in zip(tensors, functions): l = function(l, tensor)
    diff = torch.absolute(channel - l)
    cumdev = (((diff + 1) ** 2) - 1).sum()
    return cumdev


def train(channel, latent, tensors, functions, steps, rate):
    dev = evaluate(channel, latent, tensors, functions)

    tensors_best = [tensor.clone() for tensor in tensors]
    dev_best = dev.clone()

    bar = tqdm(total=steps)
    update = 1000
    for i in range(steps):
        tensors = [tensor + torch.randn_like(tensor) * rate for tensor in tensors_best]
        dev = evaluate(channel, latent, tensors, functions)
        if dev < dev_best: tensors_best, dev_best = tensors, dev
        if (i+1) % update == 0:
            bar.update(update)
            bar.set_description(f"{dev_best:.4f}")
    return tensors_best, dev_best

def sharpen(channel, latent, tensors, functions):
    rate = 10.0
    dev = evaluate(channel, latent, tensors, functions)
    bar = tqdm()
    for r in range(10):
        while True:
            change = False
            for tensor in tensors:
                for x in range(tensor.shape[0]):
                    for y in range(tensor.shape[1]):
                        tensor[x,y] += rate
                        ndev = evaluate(channel, latent, tensors, functions)
                        if ndev < dev:
                            dev = ndev
                            change = True
                            continue
                        else:
                            tensor[x,y] += rate * -2.0
                        ndev = evaluate(channel, latent, tensors, functions)
                        if ndev < dev:
                            dev = ndev
                            change = True
                            continue
                        else: tensor[x,y] += rate
            bar.update()
            bar.set_description(f"{dev:.4f}")
            if not change: break
        rate /= 10.0
    return tensors, dev



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
    (int(2e+5), 1e-2),
    (int(3e+5), 1e-3),
]

functions = [torch.add, torch.matmul]
tensors = [torch.randn([1,4]), torch.randn([4,3])]

for (steps, rate) in iterations:
    tensors, dev = train(lab, latent, tensors, functions, steps, rate)

tensors, dev = sharpen(lab, latent, tensors, functions)

print(f"###\n{tensors}\n{dev:.8f}\n###")
