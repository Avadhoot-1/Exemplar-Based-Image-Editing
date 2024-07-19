import argparse
from PIL import Image
import os 
def main(img_fol, gap_size):
    path_a = f'{img_fol}/0_0.png'
    path_b = f'{img_fol}/0_1.png'
    p = 512
    image_a = Image.open(path_a).resize((p, p))
    image_b = Image.open(path_b).resize((p, p))

    # Calculate new dimensions including gaps
    new_width = 2 * p +  gap_size  # width of a + gap + b + gap + c
    new_height = p  

    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste the images into the new image with gaps
    new_image.paste(image_a, (0, 0))
    new_image.paste(image_b, (p + gap_size, 0))
    print("concat path: ", f'{img_fol}/concat.png')
    new_image.save(f'{img_fol}/concat.png')
    print(f"Output saved to {img_fol}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate images with gaps and save to specified path.")
    parser.add_argument('--img_fol', required=True, help='Path to the images and save the output image')
    args = parser.parse_args()
    root_dir = args.img_fol
    subdirs= []
    for root, dirs, files in os.walk(root_dir):
        for subdir in dirs:
            if ('_' in subdir and 'ipynb' not in subdir):
                main(os.path.join(root_dir, subdir), 10)    

