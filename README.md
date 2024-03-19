# quickdif
Quick and easy CLI inference that just works™ for a variety of Diffusers models

## Including
  - Verified working with Stable Diffusion (1.5/2.1/XL Base), Stable Cascade/Wuerstchen, Kandinsky22, Pixart
  - Supports many common settings found in GUIs
  - Iterate over parameters like prompt or CFG
  - Some more advanced tweaks like multi-Lora, colored latents, and variance masks
  - Extremely small 1-shot script using `accelerate` for hot loading models
  - Load settings from JSON, TOML, PNG

## Not Including
  - ControlNet + other Stable Diffusion extensions
  - Other diffusion models may or may not work properly
  - Multi-stage models: DFIF Stage 2, SDXL Refiner
  - Server/API for a perpetual instance
  - Output as grid
  - 100% maximum optimization
  - 1-click installer

## Installation
You must have a recent Python version installed, at least 3.10+. In essence you just need to install `requirements.txt` for your environment of choice.

#### Basic setup
```sh
> git clone https://github.com/Beinsezii/quickdif.git
> cd ./quickdif/
```

#### Create a venv in the quickdif folder
```sh
> python3 -m venv ./venv
# alternatively, it's recommended to use the full virtualenv if you have it
> virtualenv ./venv
```

#### Install dependencies
```sh
# replace with appropriate activation script for other shells
> source ./venv/bin/activate
# It's recommended to first install torch using the recommended commands from https://pytorch.org/get-started/locally/
> pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.6 # AMD example
# finally
> pip3 install -Ur ./requirements.txt
> deactivate
```

#### Run quickdif
```sh
# See all options. Always refer to the script help over the other examples in this README
> ./quickdif.sh --help

# Run with defaults
> ./quickdif.sh "high resolution dslr photograph of pink roses in the misty rain"
# Custom model
> ./quickdif.sh -m "ptx0/terminus-xl-gamma-v1" "analogue photograph of a kitteon on the beach in golden hour sun rays"
# Single files work for Stable Diffusion
> ./quickdif.sh -m ./checkpoints/sd15/dreamshaper-v6.safetensors "colorful fantasy artwork side profile of a feminine robot in a dark cyberpunk city"
# Four dog and four cat images at twenty steps
> ./quickdif.sh "photo of a dog" "painting of a cat" -B 4 -s 20
# Colored latent for an offset noise effect
> ./quickdif.sh "underwater photograph of a dark cave full of bioluminescent glowing mushrooms" -g 9.0 -s 30 -C black -c 0.8
# Compile for a long job
> ./quickdif.sh $(cat prompts.txt) --compile
# Export favorite settings to the defaults JSON
> ./quickdif.sh -m "stabilityai/stable-cascade" -s 20 -n "blurry, noisy, cropped" --json ./quickdif.json
# Save a style to a custom JSON
> ./quickdif.sh "fantasy artwork of a kitten wearing gothic plate armor" -g 10 -G 0.5 --json ./epic_kitten.json
# Merge multiple configs
> ./quickdif.sh -I underwater_cave.png epic_kitten.json
```

## F.A.Q.
Question|Answer
---|---
Why not X popular UI?|SD.Next's diffusers backend is extremely buggy/broken in multiple areas and InvokeAI (+non-diffusers UIs) only really supports Stable Diffusion.
Windows?|The python script should work just fine but you'll need to set up the powershell/CMD stuff on your own.
Gradio?|No. If a UI ever gets made for this it'll be its own separate entity that interfaces via API. Cramming a bad gradio interface into the main script wont do anyone any favors
Feature XYZ?|Maybe. Things in the *Not Including* list may come eventually™ if this script winds up being useful enough
