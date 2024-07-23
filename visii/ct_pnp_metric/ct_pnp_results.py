from s_visual import *
# from clip_score import *
# from directional_clip import *
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    parser.add_argument("--param", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    # fol = args.fol
    scale = args.param
    for fol in os.listdir(f'/mnt/localssd/ashutosh/images/'):
        img1 = f'/mnt/localssd/ashutosh/images/{fol}/0_0.png'
        img2 = f'/mnt/localssd/ashutosh/images/{fol}/0_1.png'
        img3 = f'/mnt/localssd/ashutosh/images/{fol}/1_0.png'
        img4 = f'/mnt/localssd/ashutosh/results/{fol}/clipdiff+clipC+pnp.png'
        if not os.path.exists(img4):
            print("Not found ", fol)
            exit(0)
        s_vis_score = get_s_visual(img1, img2, img3, img4)
        data_dict = {}
        data_dict['s_visual'] = float(s_vis_score)
        with open(f'/mnt/localssd/ashutosh/images/{fol}/clipdiff+clipC+pnp.json', 'w') as file:
            json.dump(data_dict, file)

    