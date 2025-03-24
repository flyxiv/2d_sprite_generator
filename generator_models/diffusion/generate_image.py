"""Generate image based on sd1.5 and pretrained IP-Adapter

run ex) from nino base dir,
```sh
python -m generation_inpaint.generate_image --out-path ./generated_img/1.png --input-img-path .\data\generation\portrait\input.png --reference-img-path .\data\generation\sprite.png
```
"""

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from PIL import Image

import os
import torch
import argparse

def generate_image(*, out_path, input_img_path, stable_diffusion_pretrained_model, ip_adapter_pretrained_model, subfolder, weight_name, use_gpu=True, seed=42):
    device = "cuda" if use_gpu else "cpu"
    pipeline = AutoPipelineForText2Image.from_pretrained(stable_diffusion_pretrained_model, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline.load_ip_adapter(ip_adapter_pretrained_model, subfolder, weight_name=weight_name)

    input_img = load_image(input_img_path) 

    generator = torch.Generator(device).manual_seed(seed)

    generated_img = pipeline(
        prompt="a sks 2d pixel sprite of game character",
        negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality, deformed face",
        ip_adapter_image=input_img,
        generator=generator,
        strength=5,
        ip_adapter_scale=0.9,
        guidance_scale=15.0,
        num_inference_steps=300
    ).images

    os.makedirs(out_path, exist_ok=True)

    generated_img[0].show()
    generated_img[0].save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, required=True, default="generated image output path")
    parser.add_argument("--input-img-path", type=str, required=True, default="path of the base input image")

    parser.add_argument("--stable-diffusion-pretrained-model", type=str, default="sd-legacy/stable-diffusion-inpainting", required=False, help="pretrained diffusion model file")
    parser.add_argument("--ip-adapter-pretrained-model", type=str, default="h94/IP-Adapter", required=False, help="model version of IP-Adapter")
    parser.add_argument("--subfolder", type=str, default="models", required=False, help="subfolder the adapter belongs in")
    parser.add_argument("--weight-name", type=str, default="ip-adapter_sd15.bin", required=False, help="name of the weight file for the pretrained IP-Adapter")
    parser.add_argument("--seed", type=int, default=42, required=False)

    args = parser.parse_args()

    generate_image(**vars(args))

    