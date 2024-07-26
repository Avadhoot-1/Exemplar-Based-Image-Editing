
# import clipv2
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

def get_clip_score(img_fol, folder):
    imgs_path_only_ct = []
    imgs_path_only_llava = []
    imgs_path_ct_llava = []
    
    result_only_ct = {}
    result_only_llava = {}
    result_ct_llava = {}
       
    img_fol = f'{img_fol}/ins.txt'
    with open(img_fol, 'r') as file:
        s = file.read()
    print(s)
    for i in [1.5]:
        for j in [8, 10, 12]:
            a1 = []
            a2 = []
            a3 = []
            for k in range(4):
                for l in range(2):
            #         imgs_path_only_ct.append(f'{folder}/only_ct_img_{i}_cond_{j}:{k}_{l}.png')
            #         curr_img = Image.open(imgs_path_only_ct[-1])
            #         tensor_image = transforms.ToTensor()(curr_img)
            #         tensor_image_scaled = tensor_image * 255.0
            #         tensor_image_scaled = tensor_image_scaled.to(torch.uint8)
            #         score = metric(tensor_image_scaled, s)
            #         p = score.detach()
            #         a1.append(float(p))
                    
            #         imgs_path_only_llava.append(f'{folder}/only_llava_img_{i}_cond_{j}:{k}_{l}.png')
            #         curr_img = Image.open(imgs_path_only_llava[-1])
            #         tensor_image = transforms.ToTensor()(curr_img)
            #         tensor_image_scaled = tensor_image * 255.0
            #         tensor_image_scaled = tensor_image_scaled.to(torch.uint8)
            #         score = metric(tensor_image_scaled, s)
            #         p = score.detach()
            #         a2.append(float(p))
                    
            #         imgs_path_ct_llava.append(f'{folder}/ct_llava_img_{i}_cond_{j}:{k}_{l}.png')
            #         curr_img = Image.open(imgs_path_ct_llava[-1])
            #         tensor_image = transforms.ToTensor()(curr_img)
            #         tensor_image_scaled = tensor_image * 255.0
            #         tensor_image_scaled = tensor_image_scaled.to(torch.uint8)
            #         score = metric(tensor_image_scaled, s)
            #         p = score.detach()
            #         a3.append(float(p))
            # result_only_ct[f'{i}_{j}'] = sum(a1)/len(a1)
            # result_only_llava[f'{i}_{j}'] = sum(a2)/len(a2)
            # result_ct_llava[f'{i}_{j}'] = sum(a3)/len(a3)
    # print(result_only_ct)     
    with open(f'{folder}/clip_only_ct.json', 'w') as file:
        json.dump( result_only_ct, file)
    with open(f'{folder}/clip_only_llava.json', 'w') as file:
        json.dump( result_only_llava, file)
    with open(f'{folder}/clip_ct_llava.json', 'w') as file:
        json.dump( result_ct_llava, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--img_fol", type=str, help="Path to the directory containing images")
    parser.add_argument("--res_fol", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    get_clip_score(args.img_fol, args.res_fol)