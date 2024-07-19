import statistics
import argparse
import json
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import pandas as pd

root_dir = '/mnt/localssd/avadhoot/image_folders1'
subdirs= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir):
            subdirs.append(subdir)

root_dir = '../results'
subdirs1= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir and 'pear' not in subdir and 'wizard' not in subdir and '.ipynb' not in subdir and 'watercolor' not in subdir):
            subdirs1.append(subdir)

metrics = ['lpips','fid', 'hps', 'ssim', 'clip','dir_sim', 's_vis']
methods = ['only_ct', 'only_llava', 'ct_llava']

final_res = {method: {metric: [] for metric in metrics} for method in methods}

params = ['1.5_12']
print(params)
for param in params:
    for i in subdirs:
        for m in metrics:
            for d in methods:
                with open(f'/mnt/localssd/avadhoot/results/{i}/{m}_{d}.json', 'r') as file:
                    data = json.load(file)
                final_res[d][m].append(data[param])
    
    for i in subdirs1:
        for m in metrics:
            for d in methods:
                with open(f'../results/{i}/{m}_{d}.json', 'r') as file:
                    data = json.load(file)
                final_res[d][m].append(data[param])

summary_res = {}
for method, metrics_data in final_res.items():
    summary_res[method] = {}
    for metric, values in metrics_data.items():
        print(len(values))
        mean = np.mean(values)
        variance = np.var(values)
        cov = np.std(values)/np.mean(values)
        summary_res[method][metric] = (mean, variance,cov)
        
# Convert the summary dictionary to a DataFrame
data = {'Metric': metrics}
for method in methods:
    data[method] = [f"mean: {summary_res[method][metric][0]:.2f}, CoV: {summary_res[method][metric][2]:.4f}" for metric in metrics]

df = pd.DataFrame(data)

df.to_csv(f'1.5_12.csv')
# Print the DataFrame
print(df)

