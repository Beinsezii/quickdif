import argparse

parser = argparse.ArgumentParser(description='Measure cardinal colors on VAE')
parser.add_argument('-v', '--vae', type=str, default='stabilityai/sdxl-vae')
parser.add_argument('-s', '--size', type=int, default=1024)
args = parser.parse_args()

import torch, diffusers
from diffusers import image_processor

vae = diffusers.AutoencoderKL.from_pretrained(str(args.vae), use_safetensors=True)
vae.set_default_attn_processor()
processor = image_processor.VaeImageProcessor(2**(len(vae.config.block_out_channels) - 1))

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
    img = torch.tensor(pix, dtype=torch.float32).expand([1, args.size, args.size, 3]).permute((0, 3, 1, 2)).clone()
    img = processor.preprocess(img)
    tensor = vae.to('cuda').encode(img.to('cuda')).latent_dist.sample() * vae.config.scaling_factor
    results[col] = [c.mean().item() for c in tensor[0]]

    del img, tensor

print(results)
