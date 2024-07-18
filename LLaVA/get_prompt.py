from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description="Evaluate model with given image and prompt")
parser.add_argument('--img_fol', type=str, required=True, help='Path to the input folder')
parser.add_argument('--res_fol', type=str, required=True, help='Path to the result folder')
args1 = parser.parse_args()
model_path = "liuhaotian/llava-v1.6-vicuna-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)  

# Get Name of all the subdirectories
root_dir = args1.img_fol
subdirs= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir and 'ipynb' not in subdir):
            subdirs.append(subdir)

for sd in tqdm(subdirs):
    if('ipynb' in sd):
        continue
    with open(os.path.join(args1.res_fol, sd, 'edit.txt'), 'r') as file:
        inv_ins = file.read()
    prompt = f'''Give me one line edit instruction if I want to edit given input image. The edit should similar to the edit between Image1 and image2; which is: "{inv_ins}". Write your instruction in one line based on content of given input image. That is, if some part of edit between image1 and image2 is not applicable to this input image you should skip it (for example if edit between image1 and image2 is to replace a pen with pencil, but the given input image does not contain any pen; just ignore that edit). Write final instruction in one line at the end on new line. Make sure that on applying your given edit instruction on the input image, we get similar edit on given image as the edit between image1 and image2. The instruction should contain no more than 20 words so try to summarize the prompt and use only relevant information using given image. Write only one line instruction with less than 20 words. (do not mention image1, image2 in your response).'''
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
    with open(os.path.join(args1.res_fol, sd, 'inv_ins.txt'), 'w') as file:
        file.write(out)