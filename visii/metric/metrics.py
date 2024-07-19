import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
_ = torch.manual_seed(123)


from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64).to('cuda')

def fid_score(img1_path, img2_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),          
    ])  
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')   
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
    img1 = (img1 * 255).to(torch.uint8).to('cuda')
    img2 = (img2 * 255).to(torch.uint8).to('cuda')
    
    fid.update(img1, real=True)
    fid.update(img2, real=False)
    fid.update(img1, real=True)
    fid.update(img2, real=False)
    fid_score = fid.compute()
    
    return float(fid_score)

# from torchmetrics.image.inception import InceptionScore
# inception = InceptionScore()

def inception_score(img1_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),          
    ])  
    
    img1 = Image.open(img1_path).convert('RGB')   
    img1 = transform(img1).unsqueeze(0).to('cuda')
    img1 = (img1 * 255).to(torch.uint8).to('cuda')
    
    inception.update(img1)
    inc_score = inception.compute()
    
    return float(inc_score[0])

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to('cuda')

def lpips_score(img1_path, img2_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),         
        transforms.Normalize((0.5,), (0.5,))
    ]) 
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')   
    img1 = transform(img1).unsqueeze(0).to('cuda')
    img2 = transform(img2).unsqueeze(0).to('cuda')
    
    lpips_s = lpips(img1, img2)
    return float(lpips_s)
    
from torchmetrics.image import StructuralSimilarityIndexMeasure   
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')

def ssim_score(img1_path, img2_path):
    transform = transforms.Compose([
        transforms.ToTensor(),          
    ]) 
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')   
    img1 = transform(img1).unsqueeze(0).to('cuda')
    img2 = transform(img2).unsqueeze(0).to('cuda')

    ssim_s = ssim(img1, img2)
    return float(ssim_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--img1", type=str, help="Path to the directory containing images")
    parser.add_argument("--img2", type=str, help="Path to the directory containing images")
    
    args = parser.parse_args()
    fid_val = fid_score(args.img1, args.img2)
    lpips_val = lpips_score(args.img1, args.img2)
    # inception_val = inception_score(args.img2)
    ssim_val = ssim_score(args.img1, args.img2)

    print("fid: (Lower better [0 to 100]): ", fid_val)
    print("lpips: (Lower better [0 to 1]): ", lpips_val)
    # print("inception: (<3 bad, 3 -5 moderate, > 5 good)", inception_val)
    print("ssim: (range(-1 to 1) 1 is best) ", ssim_val)
    