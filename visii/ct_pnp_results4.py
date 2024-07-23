import hpsv2
import argparse
import json
import os
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    # parser.add_argument("--param", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    fol = args.fol
    # scale = args.param
    for fol in os.listdir(f'/mnt/localssd/ashutosh/images/'):
        img_fol = f'/mnt/localssd/ashutosh/images/{fol}/ins.txt'
        with open(img_fol, 'r') as file:
            s = file.read()
        img1 = f'/mnt/localssd/ashutosh/images/{fol}/0_0.png'
        img2 = f'/mnt/localssd/ashutosh/images/{fol}/0_1.png'
        img3 = f'/mnt/localssd/ashutosh/images/1_0.png'
        img41 = f'/mnt/localssd/ashutosh/results/{fol}/clipdiff+clipC+pnp.png'
        # img42 = f'/mnt/localssd/ashutosh/results/{fol}/pnpclipdiff_llava_scale_15.png'
        # img43 = f'/mnt/localssd/ashutosh/results/{fol}/pnpclipdiff_llava_scale_20.png'
        if not os.path.exists(img41):
            print("Not found ", fol)
            exit(0)
        # print((Image.open(img4)).size)
        hps_score = hpsv2.score([img41, img42, img43], s, hps_version="v2.1") 
        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_10.json', 'r') as file:
            data_dict = json.load(file)
        data_dict['hps_score'] = float(hps_score[0])
        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_10.json', 'w') as file:
            json.dump(data_dict, file)  

        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_15.json', 'r') as file:
            data_dict = json.load(file)
        data_dict['hps_score'] = float(hps_score[1])
        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_15.json', 'w') as file:
            json.dump(data_dict, file)  

        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_20.json', 'r') as file:
            data_dict = json.load(file)
        data_dict['hps_score'] = float(hps_score[2])
        with open(f'/mnt/localssd/ashutosh/images/{fol}/ct_pnp_20.json', 'w') as file:
            json.dump(data_dict, file)  


    