# from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import statistics
import argparse
import json
from tqdm import tqdm
from PIL import Image
import os
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
device = 'cuda'
class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two,text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two)
        return sim_direction

    def forward(self, image_one, image_two, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        # text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two,  text_feat_two
        )
        return directional_similarity
clip_id = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)
dir_similarity = DirectionalSimilarity(tokenizer, text_encoder, image_processor, image_encoder)
# root_dir = './results'
# subdirs= []
# for root, dirs, files in os.walk(root_dir):
#     for subdir in dirs:
#         if ('_' in subdir and 'pear' not in subdir and 'wizard' not in subdir and '.ipynb' not in subdir and 'watercolor' not in subdir):
#             subdirs.append(subdir)
ct_scores_all = {}
llava_scores_all = {}
ct_llava_scores_all = {}
snt = 0

def get_dir_score(img_fol, res_fol):
    ct_scores = {}
    llava_scores = {}
    ct_llava_scores = {}
    img1 = f'{img_fol}/1_0.png'
    with open(f'{img_fol}/ins.txt', 'r') as file:
        s = file.read() 
    for z in [1.5]:
        for j in [8,10,12]:
            a1 = []
            a2 = []
            a3 = []
            for k in range(4):
                for l in range(2):
                    img2 = f'{res_fol}/only_ct_img_{z}_cond_{j}:{k}_{l}.png'
                    a1.append(dir_similarity(Image.open(img1), Image.open(img2), s).item())
                    img2 = f'{res_fol}/only_llava_img_{z}_cond_{j}:{k}_{l}.png'
                    a2.append(dir_similarity(Image.open(img1), Image.open(img2), s).item())
                    img2 = f'{res_fol}/ct_llava_img_{z}_cond_{j}:{k}_{l}.png'
                    a3.append(dir_similarity(Image.open(img1), Image.open(img2), s).item())
            ct_scores[f'{z}_{j}'] = (sum(a1)/len(a1))
            llava_scores[f'{z}_{j}'] = (sum(a2)/len(a2))
            ct_llava_scores[f'{z}_{j}'] = (sum(a3)/len(a3))
    with open(f'{res_fol}/dir_sim_only_ct.json', 'w') as file:
        json.dump(ct_scores, file)
    with open(f'{res_fol}/dir_sim_only_llava.json', 'w') as file:
        json.dump(llava_scores, file)
    with open(f'{res_fol}/dir_sim_ct_llava.json', 'w') as file:
        json.dump(ct_llava_scores, file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--img_fol", type=str, help="Path to the directory containing images")
    parser.add_argument("--res_fol", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    get_dir_score(args.img_fol, args.res_fol)