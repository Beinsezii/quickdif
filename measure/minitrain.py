#! /usr/bin/python
import torch
from tqdm import tqdm
import json
import argparse

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_grad_enabled(False)

def evaluate(channel, latent, m, a):
    l = latent.reshape([21, 21, 21, 4, 1])
    table = l.expand([21, 21, 21, 4, 4]).swapaxes(-1,-2).clone().mul(l)
    table += a
    table *= m
    adjusted = torch.sum(table, axis=[-1,-2])
    diff = (channel - adjusted)
    cumdev = torch.absolute(diff ** 2).sum()
    meandev = diff.mean()
    return cumdev, meandev


def train(channel, latent, steps, rate):
    mul = torch.randn([4, 4])
    add = torch.randn([4, 4])

    dev, mean = evaluate(channel, latent, mul, add)

    mul_best = mul.clone()
    add_best = add.clone()
    dev_best = dev.clone()
    mean_best = mean.clone()

    bar = tqdm(total=steps)
    repeat = 0
    for i in range(steps):
        if dev < dev_best: mul_best, add_best, dev_best, mean_best = mul.clone(), add.clone(), dev.clone(), mean.clone()
        if repeat > 32:
            mul = mul_best + torch.randn([4, 4]) * rate * 100
            add = add_best + torch.randn([4, 4]) * rate * 100
            dev, mean = evaluate(channel, latent, mul, add)
        for tensor in [add, mul]:
            for x in range(4):
                for y in range(4):
                    same = False
                    tensor[x,y] += rate
                    ndev, nmean = evaluate(channel, latent, mul, add)
                    if ndev < dev:
                        dev, mean = ndev, nmean
                    else:
                        tensor[x,y] += rate * -2
                        ndev, nmean = evaluate(channel, latent, mul, add)
                        if ndev < dev:
                            dev, mean = ndev, nmean
                        else:
                            tensor[x,y] += rate
                            same = True
                    if same: repeat += 1
                    else: repeat = 0
        bar.update()
        bar.set_description(f"{dev_best} / {mean_best} | {dev} / {mean}")
    return add_best, mul_best, dev_best, mean_best


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

steps = 10000
rate = 1e-3

l_add, l_mul, l_dev, l_mean = train(lab[:,:,:,0], latent, steps, rate)
a_add, a_mul, a_dev, a_mean = train(lab[:,:,:,1], latent, steps, rate)
b_add, b_mul, b_dev, b_mean = train(lab[:,:,:,2], latent, steps, rate)

print(f"### L ###\n+ {l_add}\nX {l_mul}\n{l_dev} / {l_mean}\n### L ###")
print(f"### A ###\n+ {a_add}\nX {a_mul}\n{a_dev} / {a_mean}\n### A ###")
print(f"### B ###\n+ {b_add}\nX {b_mul}\n{b_dev} / {b_mean}\n### B ###")
