from directional_clip import *
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    parser.add_argument("--param", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    fol = args.fol
    scale = args.param
    img_fol = f'/mnt/localssd/ashutosh/images/{fol}/ins.txt'
    with open(img_fol, 'r') as file:
        s = file.read()
    img1 = f'/mnt/localssd/ashutosh/images/{fol}/0_0.png'
    img2 = f'/mnt/localssd/ashutosh/images//{fol}/0_1.png'
    img3 = f'/mnt/localssd/ashutosh/images/{fol}/1_0.png'
    img4 = f'/mnt/localssd/ashutosh/results/{fol}/clipdiff+clipC+pnp.png'
    if not os.path.exists(img4):
        print("Not found ", fol)
        exit(0)
    dir_sim_score = dir_similarity(Image.open(img3), Image.open(img4), s).item()
    with open(f'/mnt/localssd/ashutosh/images/{fol}/clipdiff+clipC+pnp.json', 'r') as file:
        data_dict = json.load(file)
    data_dict['dim_sim_score'] = dir_sim_score
    with open(f'/mnt/localssd/ashutosh/images/{fol}/clipdiff+clipC+pnp.json', 'w') as file:
        json.dump(data_dict, file)  


    