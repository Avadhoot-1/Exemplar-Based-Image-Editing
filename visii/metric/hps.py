import argparse
import hpsv2
import json
from PIL import Image
def get_hps_score(folder):

    img_fol = folder.split('/')[-1]     
    img_fol = f'/mnt/localssd/avadhoot/image_folders1/{img_fol}/ins.txt'
    with open(img_fol, 'r') as file:
        s = file.read()
    ct_dict = {}
    llava_dict = {}
    ct_llava_dict = {}
    for i in [1.5]:
        for j in [8, 10, 12]:
            imgs_path_only_ct = []
            imgs_path_only_llava = []
            imgs_path_ct_llava = []
            for k in range(4):
                for l in range(2):
                    imgs_path_only_ct.append(f'{folder}/only_ct_img_{i}_cond_{j}:{k}_{l}.png')
                    imgs_path_only_llava.append(f'{folder}/only_llava_img_{i}_cond_{j}:{k}_{l}.png')
                    imgs_path_ct_llava.append(f'{folder}/ct_llava_img_{i}_cond_{j}:{k}_{l}.png')


            result_only_ct = hpsv2.score(imgs_path_only_ct, s, hps_version="v2.1") 
            result_only_llava = hpsv2.score(imgs_path_only_llava, s, hps_version="v2.1")
            result_ct_llava = hpsv2.score(imgs_path_ct_llava, s, hps_version="v2.1")

            result_only_ct = [float(p) for p in result_only_ct]
            result_only_llava = [float(p) for p in result_only_llava]
            result_ct_llava = [float(p) for p in result_ct_llava]
            average_hps_only_ct = sum(result_only_ct)/len(result_only_ct)
            average_hps_only_llava = sum(result_only_llava)/len(result_only_llava )
            average_hps_ct_llava = sum(result_ct_llava)/len(result_ct_llava )
            ct_dict[f'{i}_{j}'] = average_hps_only_ct
            llava_dict[f'{i}_{j}'] = average_hps_only_llava
            ct_llava_dict[f'{i}_{j}'] = average_hps_ct_llava
    
    # hps_scores_only_ct = dict(zip(imgs_path_only_ct, result_only_ct))
    # hps_scores_only_llava = dict(zip(imgs_path_only_llava, result_only_llava))
    # hps_scores_ct_llava = dict(zip(imgs_path_ct_llava, result_ct_llava))
    
    # sorted_items_only_ct = sorted(hps_scores_only_ct.items(), key=lambda item: item[1], reverse=True)
    # sorted_items_only_llava = sorted(hps_scores_only_llava.items(), key=lambda item: item[1], reverse=True)
    # sorted_items_ct_llava = sorted(hps_scores_ct_llava.items(), key=lambda item: item[1], reverse=True)
    
    # top_3_only_ct = [[item[0], hps_scores_only_ct[item[0]] ] for item in sorted_items_only_ct[:3]]
    # top_3_only_llava = [[item[0], hps_scores_only_llava[item[0]] ] for item in sorted_items_only_llava[:3]]
    # top_3_ct_llava = [[item[0], hps_scores_ct_llava[item[0]] ] for item in sorted_items_ct_llava[:3]]

    # Create the dictionary
    # variables_dict = {
    #     'average_hps_only_ct': average_hps_only_ct,
    #     'average_hps_only_llava': average_hps_only_llava,
    #     'average_hps_ct_llava': average_hps_ct_llava,
    #     'top_3_only_ct': top_3_only_ct,
    #     'top_3_only_llava': top_3_only_llava,
    #     'top_3_ct_llava': top_3_ct_llava
    # }
    with open(f'{folder}/hps_only_ct.json', 'w') as file:
        json.dump(ct_dict, file)
    with open(f'{folder}/hps_only_llava.json', 'w') as file:
        json.dump(llava_dict, file)
    with open(f'{folder}/hps_ct_llava.json', 'w') as file:
        json.dump(ct_llava_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    get_hps_score(args.fol)