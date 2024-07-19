import os
import argparse
from PIL import Image

def resize_image_inplace(image_path, size=(512, 512)):
    image_a = Image.open(image_path).resize((512, 512))
    image_a = image_a.convert('RGB')
    image_a.save(image_path)

def process_directory(directory, size=(512, 512)):
    # print("aba")
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory")
        return
    
    for filename in os.listdir(directory):
        print(directory)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            image_path = os.path.join(directory, filename)
            resize_image_inplace(image_path, size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--directory", type=str, help="Path to the directory containing images")
    
    args = parser.parse_args()
    root_dir = args.directory
    subdirs= []
    for root, dirs, files in os.walk(root_dir):
        for subdir in dirs:
            if ('_' in subdir and 'ipynb' not in subdir):
                process_directory(os.path.join(root_dir, subdir))
