from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
from tqdm import tqdm
import argparse
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Define command-line arguments
parser = argparse.ArgumentParser(description="Evaluate model with given image and prompt")
parser.add_argument('--img_fol', type=str, required=True, help='Path to the input image')
parser.add_argument('--res_fol', type=str, required=True, help='Path to the directory to save the output')
args = parser.parse_args()
model_path = "liuhaotian/llava-v1.6-vicuna-13b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

root_dir = args.img_fol
subdirs= []
for root, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        if ('_' in subdir and 'ipynb' not in subdir):
            subdirs.append(subdir)
for sd in tqdm(subdirs):            
    prompt = "Given image grid contains two images image1 on left and image2 on right. image2 is edited version of image1. Explain what edits are done on image1 to get image2. These edits can include addition or removal of objects, One object changing into some other object, change of style, etc. explain the edits in detail. Describe the edits only, for the things which are not changes or edited do not mention them, ignore the minor changes, focus on edit at broader level. Give your answer in less than 100 words in a single paragraph (Do not give numbered list). Your edits should be such that using that information on image1 we should be able to generate image2"

    # Load image path from command-line argument
    image_file = os.path.join(args.img_fol, sd, 'concat.png')
    
    # Create arguments object for evaluation
    eval_args = type('Args', (), {
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

    out = eval_model(eval_args, tokenizer, model, image_processor, context_len)
    # Write the output to a file
    with open(os.path.join(args.res_fol, sd, 'edit.txt'), 'w') as file:
        file.write(out)

