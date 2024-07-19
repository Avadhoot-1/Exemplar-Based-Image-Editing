# Exemplar-Based-Image-Editing
Intern Project @Adobe

## setup
1. Install visii env
```Shell
   conda create -n visii python=3.10
   conda activate visii
   pip install -r requirements.txt
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   pip uninstall charset-normalizer
   pip install ftfy
   pip install sentence_transformers
   pip install tensorboard
   pip install git+https://github.com/huggingface/accelerate {Then restart the kernel}
   ```
2. Install LLaVA env
```bash
   cd LLaVA
   conda create -n llava python=3.10 -y
   conda activate llava
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   pip install protobuf
   pip install torchmetrics[image]
   ```
3. Install HPS env
```bash
   git clone https://github.com/tgxs002/HPSv2.git
   cd HPSv2
   pip install -e . 
   ```
## Instructions to run the Project 

Maintain the directory structure as follows:
```
{image_folder}
└───{subfolder1}
|    │   0_0.png # before image
|    │   0_1.png # after image
|    │   1_0.png # test image
|
└───{subfolder2}
    │   0_0.png # before image
    │   0_1.png # after image
    │   1_0.png # test image
```
```
{result_folder}
└───{subfolder1}
|    │   results will be stored here
|
└───{subfolder2}
    │   results will be stored here
```
**Note: Names of the subfolder must contain underscore ('_') **
Then run the following commands:
- Resize all the images to 512*512
  
 ```
python3 preprocess.py --directory [Path to image_folder]
```

- make an image concat.png in each exemplar folder
  
 ```
python3 concat.py --img_fol [Path to image_folder]
```

## Obtaining LLaVA Instruction/ Caption
```
cd LLaVA
```
- The following command will make file 'edit.txt' for each subfolder in `image_folder` and this file will be stored in corrosponding subfolder in `result_folder`

```
python3 edit.py --img_fol [absolute path of image folder] --res_fol [absolute path to result folder]
```

- After running above command run follwing command to get 'inv_ins.txt' {LLaVA instruction [Not caption]} for each subfolder in image_folder; the file will be stored in corrosponding subfolder in results folder.

```
python3 get_prompt.py --img_fol [absolute path of image folder] --res_fol [absolute path to result folder]
```

- Similarly, to get caption {For pnp method} for all subfolders run follwing command; this will produce files inv_cap.txt.

```
python3 get_caption.py --img_fol [absolute path of image folder] --res_fol [absolute path to result folder]
```

- After getting all inv_ins.txt or inv_cap.txt run the following commands which Truncate the sentence to the maximum number of words less than or equal to 40 by finding the last full stop within the first 40 words.
```
python3 truncate_prompt.py --res_fol [path to result folder]
```
```
python3 truncate_caption.py --res_fol [path to result folder]
```

## Running LLaVA pipeline
- The file `llava_pipeline.sh` performs the CT optimization (Train) and run all 3 methods Visii, LLaVA and CT + LLaVA for Img_CFG:1.5 and  3 different text_CFGs: [8,10,12]. This script also calculates s_visual score of the images.

```
./llava_pipeline.sh [Name of subdirectory] [Path to log folder] [Path to result folder] [Path to image folder]
```
For example:
```
./llava_pipeline.sh add_crown ./logs ./results ./images
```

- If you have to run the above script for all subfolders in image directory you can use the following command, which will run above script on all the subfolders in image folder

```
./superscript.sh [Path to log folder] [Path to result folder] [Path to image folder]
```

## TO obtain LLaVA + Caption Embeddings {For PnP}
run the follwing script; This script will produce the file subfolder_llava+ct.pt in the folder PT_folder for all subfolders in image_folder.

```bash
./superscript.sh [Path to log folder {To get CT}] [Path to result folder {To get file inv_cap.txt}] [Path to image folder] [Path to PT_folder]
```

## The Result Folder:
# Images

File: ```ct_llava_img_1.5_cond_8:0_0.png``` means that the method used is CT + LLaVA; Image CFG is 1.5; text CFG is 8. and since for each hyperparameter 8 images are generated; these images are represented by 0_0, 0_1, 1_0, 1_1, 2_0, 2_1, 3_0, 3_1.

Relevant Parameters:
```python
  method: ['only_ct', 'only_llava', 'ct_llava']
  IMG_CFG: [1.5]
  Text_CFG: [8, 10, 12]
  ```
   




