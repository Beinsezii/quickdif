#! /usr/bin/python
import torch
from tqdm import tqdm
import json
import argparse

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.set_printoptions(precision=12)

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

srgb = torch.tensor(list(map(
    lambda L: list(map(
        lambda A: list(map(
            lambda B: B['SRGB'],
            A
            )),
        L
        )),
    data['data']
    )))

del data

tensors = [
    torch.nn.Parameter(torch.randn([4, 3])),
    torch.nn.Parameter(torch.zeros([1, 3])),
]
functions = [
    torch.matmul,
    torch.add,
]

optimizer = torch.optim.AdamW(tensors, lr=1e-3)

source = torch.flatten(latent, end_dim=2)
target = torch.flatten(srgb, end_dim=2)

steps = int(1e+5)
update = 100
bar = tqdm(total=steps)

try:
    for epoch in range(steps):
        prediction = source.clone()
        for (t, f) in zip(tensors, functions): prediction = f(prediction, t)
        loss = torch.absolute((prediction - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % update == 0:
            bar.update(update)
            bar.set_description(f"Epoch {epoch+1} Loss {loss:.4f}")
except KeyboardInterrupt:
    print()

print("###\n")
for tensor in tensors: print(f"{tensor}\n")
print("###")
