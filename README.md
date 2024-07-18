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

## Obtaining LLaVA Instruction/ Caption
For the directory structure
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





