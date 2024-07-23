from metrics import *
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    parser.add_argument("--param", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    fol = args.fol
    scale = args.param
    for fol in os.listdir(f'/mnt/localssd/ashutosh/images/'):
        img1 = f'/mnt/localssd/ashutosh/images/{fol}/0_0.png'
        img2 = f'/mnt/localssd/ashutosh/images/{fol}/0_1.png'
        img3 = f'/mnt/localssd/ashutosh/images/{fol}/1_0.png'
        img4 = f'/mnt/localssd/ashutosh/results/{fol}/clipdiff+clipC+pnp.png'
        if not os.path.exists(img4):
            exit(0)

        img_fol = f'/mnt/localssd/ashutosh/images/{fol}/ins.txt'
        with open(img_fol, 'r') as file:
            s = file.read()
        curr_img = Image.open(img4)
        tensor_image = transforms.ToTensor()(curr_img)
        tensor_image_scaled = tensor_image * 255.0
        tensor_image_scaled = tensor_image_scaled.to(torch.uint8)
        score = metric(tensor_image_scaled, s)
        p = score.detach()
        clip_s = float(p)
        fid_val = fid_score(img3, img4)
        lpips_val = lpips_score(img3, img4)
        # inception_val = inception_score(args.img2)
        ssim_val = ssim_score(img3, img4)

        with open(f'/mnt/localssd/ashutosh/images/{fol}/clipdiff+clipC+pnp.json', 'r') as file:
            data_dict = json.load(file)
        data_dict['clip_s'] = clip_s
        data_dict['fid_val'] = fid_val
        data_dict['lpips_val'] = lpips_val
        data_dict['ssim_val'] = ssim_val
        with open(f'/mnt/localssd/ashutosh/images/{fol}/clipdiff+clipC+pnp.json', 'w') as file:
            json.dump(data_dict, file)   
