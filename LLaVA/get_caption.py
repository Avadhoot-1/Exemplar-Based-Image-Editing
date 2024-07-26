from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse
import os
from tqdm import tqdm
import warnings
import numpy as np
import time

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description="Evaluate model with given image and prompt")
parser.add_argument('--img_fol', type=str, required=True, help='Path to the input image')
parser.add_argument('--res_fol', type=str, required=True, help='Path to the input image')
args1 = parser.parse_args()
model_path = "liuhaotian/llava-v1.6-vicuna-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)  

# Get name of all subdirectories
root_dir = args1.img_fol
subdirs= [
        "sor_fox",
        "poc_rose",
        "style_woodcarving",
        "bg_fog",
        "style_palm",
        "sor_mountain2desert",
        "style_skeleton",
        "sor_leopard",
        "poc_skeleton",
        "sor_cowboy"
    ]
# for root, dirs, files in os.walk(root_dir):
#     for subdir in dirs:
#         if ('_' in subdir and 'ipynb' not in subdir):
#             subdirs.append(subdir)

print(subdirs)
time_taken = []
for sd in tqdm(subdirs):
    start_time = time.time()
    with open(os.path.join(args1.res_fol, sd, 'edit.txt'), 'r') as file:
        inv_ins = file.read()
    prompt = f'''Give me one line description of an image generated after applying the edits between Image1 and image2 on input image. The edits between image1 and image2 are: "{inv_ins}". Write your caption in one line based on content of given input image. That is, if some part of edit between image1 and image2 is not applicable to this input image you should skip it (for example if edit between image1 and image2 is to replace a pen with pencil, but the given input image does not contain any pen; just ignore that edit). Make sure that your caption completely describe the edited image obtained on editing the input image. The caption should not contain more than 20 words. so try to summarize the caption and use only relevant information using given image. Write only one line caption with less than 20 words. Write output as if you are describing the image but in one line in less than 20 words. (do not mention image1, image2 in your response).'''
    
    image_file = os.path.join(args1.img_fol, sd, '1_0.png')

    args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
    })()

    out = eval_model(args,tokenizer, model, image_processor, context_len)
    # print(out)
    with open(os.path.join(args1.res_fol, sd, 'inv_cap.txt'), 'w') as file:
        file.write(out)
    time_taken.append(time.time() - start_time)
print("Average time taken for each subfolder: ", np.mean(time_taken))
print("Variance of time taken for each subfolder: ", np.var(time_taken))