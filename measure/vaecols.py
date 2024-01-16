import torch, diffusers, argparse, json
from tqdm import tqdm
from diffusers import image_processor

parser = argparse.ArgumentParser(description='Measure cardinal colors on VAE')
parser.add_argument('-v', '--vae', type=str, default='stabilityai/sdxl-vae')
parser.add_argument('json', type=argparse.FileType(mode='w'), help="Output map as .json")
args = parser.parse_args()

dtype = torch.float32
device = 'cuda'
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')

vae = diffusers.AutoencoderKL.from_pretrained(str(args.vae), use_safetensors=True, torch_dtype=dtype).to(device)
processor = image_processor.VaeImageProcessor(2**(len(vae.config.block_out_channels) - 1))

iters = torch.combinations(torch.linspace(0, 1, 20), r=3, with_replacement=True)
iters = iters.reshape([iters.shape[0], 1, 1, iters.shape[1]]
            ).expand([iters.shape[0], vae.config.sample_size, vae.config.sample_size, iters.shape[1]])

data = []

for img in tqdm(iters):
    result = {'rgb': img[0,0].tolist()}
    # needs channels first
    tensor = vae.encode(
        processor.preprocess(img.permute(2,0,1))
    ).latent_dist.sample().permute(1,0,2,3).flatten(start_dim=1)

    result |= {
        'latent_mean' : tensor.mean(dim=1).tolist(),
        'latent_dist' : tensor.quantile(torch.linspace(0,1,51), dim=1).tolist(),
    }

    data.append(result)

json.dump({'vae_factor': vae.config.scaling_factor, 'data': data}, args.json)
