import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Measure cardinal colors on VAE')
parser.add_argument('vae', type=Path)
parser.add_argument('-s', '--size', type=int, default=1024)
args = parser.parse_args()

# assert str(args.vae).endswith('.safetensors')
# assert args.vae.is_file()

import torch, diffusers

vae = diffusers.AutoencoderKL.from_pretrained(str(args.vae), use_safetensors=True)
print(vae.config)

results = {}

for col, pix in [
        ("black",   (0.0, 0.0, 0.0)),
        ("white",   (1.0, 1.0, 1.0)),
        ("red",     (1.0, 0.0, 0.0)),
        ("green",   (0.0, 1.0, 0.0)),
        ("blue",    (0.0, 0.0, 1.0)),
        ("cyan",    (0.0, 1.0, 1.0)),
        ("magenta", (1.0, 0.0, 1.0)),
        ("yellow",  (1.0, 1.0, 0.0)),
]:
    img = torch.tensor(pix, dtype=torch.float32).expand([1, args.size, args.size, 3]).permute((0, 3, 1, 2)).clone()
    tensor = vae.to('cuda').encode(img.to('cuda'), False)[0].sample().to('cpu').type(torch.float64)
    wh = tensor.shape[-1] * tensor.shape[-2]

    results[col] = [c.sum().div(wh).item() for c in tensor[0]]

    del img, tensor


print(results)
