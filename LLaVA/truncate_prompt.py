import argparse
import os

def prompt_mod(file_path):
    with open(file_path, 'r') as file:
        paragraph = file.read().strip()
    words = paragraph.split()
    if(len(words) < 40):
        return
    prev = 0

    truncated_paragraph = ""
    word_count = 0
    max_truncated_paragraph = ""
    
    for i, word in enumerate(words):
        truncated_paragraph += word + " "
        word_count += 1
        
        if word.endswith('.'):
            max_truncated_paragraph = truncated_paragraph.strip()
        
        if word_count >= 40:
            break

    with open(file_path, 'w') as file:
        file.write(max_truncated_paragraph)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing subdirectories of images")
    args = parser.parse_args()
    root_dir = args.fol
    subdirs= []
    for root, dirs, files in os.walk(root_dir):
        for subdir in dirs:
            if ('_' in subdir and 'ipynb' not in subdir):
                if(os.path.exists(os.path.join(root_dir, subdir, 'inv_ins.txt'))):
                    prompt_mod(os.path.join(root_dir, subdir, 'inv_ins.txt'))