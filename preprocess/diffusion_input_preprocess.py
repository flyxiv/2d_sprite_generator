"""Augment image size to 

Puts image to the center, and pads image with blank background to make 512x512 image

run example: in nino base directory,
python -m diffusion_input_preprocess --image-dir ./data/input --output-dir ./data/output
"""

import argparse
import os
from pathlib import Path
from PIL import Image

DIFFUSION_IMG_SIZE = (512, 512)
TRANSPARENT_COLOR = (0, 0, 0, 0)

def pad_image_to_diffusion_input_size(img_dir, output_dir):
    img_dir_path = Path(image_dir)
    imgs = path.glob('**/*')
    
    for img_path in imgs:
        img_name = img_path.name
        img = Image.open(img_path)

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        new_img_with_correct_size = Image.new('RGBA', DIFFUSION_IMG_SIZE, TRANSPARENT_COLOR)

        new_img_center_x = (DIFFUSION_IMG_SIZE[0] / 2 - img.width)
        new_img_center_y = (DIFFUSION_IMG_SIZE[1] / 2 - img.height)

        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args('--img-dir', type=str, default=True, help='directory with images to be preprocessed.')
    parser.parse_args('--output-dir', type=str, default=True, help='directory where preprocessed image files will be saved')

    args = parser.parse_args()
    
    pad_image_to_diffusion_input_size(img_dir=args.img_dir, output_dir=args.output_dir)
