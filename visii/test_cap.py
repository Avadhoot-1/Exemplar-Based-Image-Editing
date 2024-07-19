import argparse
import glob
import json
import os

import numpy as np
import PIL
import requests
import torch
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from visii_cap import StableDiffusionVisii


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_number', type=str, default='best')
    parser.add_argument('--log_folder', type=str, default='ip2p_painting1_0_0.png')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--image_folder', type=str, default='./images')
    parser.add_argument('--pt_fol', type=str, default='./results')
    parser.add_argument('--res_fol', type=str, default='./results')

    parser.add_argument('--number_of_row', type=int, default=4)
    parser.add_argument('--number_of_col', type=int, default=2)

    parser.add_argument('--guidance_scale', type=int, default=8)
    parser.add_argument('--img_guidance_scale', type=float, default=1.5)
    parser.add_argument('--prompt', type=str, default='a husky')
    parser.add_argument('--hybrid_ins', type=bool, default=False)
    parser.add_argument('--ct', type=int, choices=[0, 1], default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionVisii.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image_folder = args.image_folder

    list_images = glob.glob(os.path.join(image_folder, '*_0.png')) + glob.glob(os.path.join(image_folder, '*_0.jpg'))

    log_dir = os.path.join(args.log_path, args.log_folder)
    os.makedirs('results', exist_ok=True)
    list_images.sort()
    for img_path in list_images[1:2]:
        before_image = Image.open(img_path).convert("RGB").resize((512, 512))
        location = os.path.join('results/{}_{}.png'.format(args.log_folder.split('.')[0], img_path.split('/')[-1]))
        checkpoint = os.path.join(log_dir, 'prompt_embeds_{}.pt'.format(args.checkpoint_number))
        opt_embs = torch.load(checkpoint)

        after_images = []
        for i in range(args.number_of_row):
            if args.hybrid_ins:
                with open(os.path.join(log_dir, 'learned_prompt.txt')) as f:
                    init_prompt = f.read()

                after_image = pipe.test_concatenate(
                    res_fol = args.pt_fol,
                    fol_name = (args.image_folder).split('/')[-1],
                    prompt_embeds=opt_embs,
                    ct=args.ct,
                    image=before_image,
                    image_guidance_scale=args.img_guidance_scale,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=20,
                    prompt=args.prompt,
                    init_prompt=init_prompt,
                    num_images_per_prompt=args.number_of_col,
                ).images
            else:
                after_image = pipe.test(
                    prompt_embeds=opt_embs,
                    image=before_image,
                    image_guidance_scale=args.img_guidance_scale,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=20,
                    num_images_per_prompt=args.number_of_col,
                ).images
            
            for index in range(len(after_image)):
                subfol = (args.image_folder).split('/')[-1]
                after_image[index].save(f'{args.res_fol}/{subfol}/{i}_{index}.png')
            
            after_images.extend(after_image)

        location = os.path.join(f'''{args.res_fol}/{subfol}/{args.log_folder.split('.')[0]}_{img_path.split('/')[-1]}.png''')
        image_grid(after_images, args.number_of_col, args.number_of_row).convert('RGB').save(location)
        print(location)
