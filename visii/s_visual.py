import inspect
import os
import random
import time
from typing import Callable, List, Optional, Union
import json
import bitsandbytes as bnb
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from PIL import Image, ImageOps

from tqdm import tqdm
from transformers import (AutoProcessor, CLIPFeatureExtractor,
                          CLIPImageProcessor, CLIPModel, CLIPTextModel,
                          CLIPTokenizer, CLIPVisionModel)


def text_projection(cond_images, target_images, device='cuda'):
    if(True):
        with torch.no_grad():
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            clip_model.eval().to(device)
            
            cond_inputs = processor(images=cond_images, return_tensors="pt").to(device)
            pooled_features_cond = clip_model.vision_model(cond_inputs['pixel_values'])[1]
            pooled_features_cond = clip_model.visual_projection(pooled_features_cond)

            target_inputs = processor(images=target_images, return_tensors="pt").to(device)
            pooled_features_target = clip_model.vision_model(target_inputs['pixel_values'])[1]
            pooled_features_target = clip_model.visual_projection(pooled_features_target)
            edit_direction_embed = pooled_features_target.mean(dim=0) - pooled_features_cond.mean(dim=0)
            edit_direction_embed = edit_direction_embed / edit_direction_embed.norm(p=2, dim=-1, keepdim=True)

            return edit_direction_embed
def get_s_visual(img1, img2, img3, img4):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img3 = Image.open(img3)
    img4 = Image.open(img4)
    dir1 = text_projection(img1,img2)
    # print(dir1.shape)
    dir2 = text_projection(img3, img4)
    # print(dir2.shape)
    criterion_cosine = nn.CosineSimilarity(dim=0)
    s_vis = criterion_cosine(dir1, dir2)
    return s_vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate images from two folders and save the result.")
    parser.add_argument('--fol1', type=str, help='Path to the first folder containing smaller images')
    parser.add_argument('--fol2', type=str, help='Path to the first folder containing smaller images')
    parser.add_argument('--res_file', type=str, help='Path to the first folder containing smaller images')
    parser.add_argument('--pref', type=str, help='Path to the first folder containing smaller images')
    parser.add_argument('--save_path', type=str, help='Path to the first folder containing smaller images')
    args = parser.parse_args()

    res= {}
    for i in range(4):
        im1 = f'{args.fol1}/0_0.png'
        im2 = f'{args.fol1}/0_1.png'
        im3 = f'{args.fol1}/1_0.png'
        img1 = Image.open(im1)
        img2 = Image.open(im2)
        img3 = Image.open(im3)
        dir1 = text_projection(img1,img2)
        for j in range(2):
            im4 = f'{args.fol2}/{i}_{j}.png'
            img4 = Image.open(im4)
            img4.save(f'{args.pref}:{i}_{j}.png')
            dir2 = text_projection(img3, img4)
            criterion_cosine = nn.CosineSimilarity(dim=0)
            s_vis = criterion_cosine(dir1, dir2)
            res[f'{i}_{j}'] = float(s_vis)
    max_key = max(res, key=res.get)
    res['best'] =  max_key
    # best_img = Image.open(f'{args.fol2}/{max_key}.png')
    # best_img.save(args.save_path)
    with open(args.res_file, 'w') as file:
        json.dump(res, file)
    