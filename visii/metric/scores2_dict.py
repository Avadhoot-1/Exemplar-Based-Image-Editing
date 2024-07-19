import statistics
import json
from tqdm import tqdm
from PIL import Image
import os

root_dir = '/mnt/localssd/avadhoot/image_folders1'
subdirs= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir):
            subdirs.append(subdir)
ct_scores_all = {}
llava_scores_all = {}
ct_llava_scores_all = {}
snt = 0


def get_avg_ind(d, ind):
    total_sum_except_best = sum(value[ind] for key, value in d.items() if key != 'best')
    return total_sum_except_best/(len(d))   
    
def get_avg(d):
    total_sum_except_best = sum(value for key, value in d.items() if key != 'best')
    return total_sum_except_best/(len(d) -1)

for i in tqdm(subdirs, desc="Processing subdirectories"):
    s_vis_ct_scores = {}
    s_vis_llava_scores = {}
    s_vis_ct_llava_scores = {}
    
    for z in [1.5]:
        for j in [8,10,12]:

            with open(f'/mnt/localssd/avadhoot/results/{i}/only_ct_img_{z}_cond_{j}.json', 'r') as f:
                only_ct = json.load(f)  
            s_vis_ct_scores[f'{z}_{j}'] = get_avg(only_ct) 
            
            with open(f'/mnt/localssd/avadhoot/results/{i}/only_llava_img_{z}_cond_{j}.json', 'r') as f:
                only_llava = json.load(f)  
            s_vis_llava_scores[f'{z}_{j}'] = get_avg(only_llava)   
            
            with open(f'/mnt/localssd/avadhoot/results/{i}/ct_llava_img_{z}_cond_{j}.json', 'r') as f:
                ct_llava = json.load(f)  
            s_vis_ct_llava_scores[f'{z}_{j}'] = get_avg(ct_llava) 
            
    dicts_to_save = {
    's_vis_only_ct.json': s_vis_ct_scores,
    's_vis_only_llava.json': s_vis_llava_scores,
    's_vis_ct_llava.json': s_vis_ct_llava_scores,
    }

    for filename, dictionary in dicts_to_save.items():
        with open(f'/mnt/localssd/avadhoot/results/{i}/{filename}', 'w') as json_file:
            json.dump(dictionary, json_file)