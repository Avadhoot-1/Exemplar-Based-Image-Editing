from metrics import *
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--fol", type=str, help="Path to the directory containing images")
    args = parser.parse_args()
    fol = args.fol
    metric = ['fid', 'lpips', 'ssim']
    method = ['only_ct', 'only_llava', 'ct_llava']
    for m in metric:
        for d in method:
            res = {}
            for i in [1.5]:
                for j in [8, 10, 12]:
                    a1 = []
                    for k in range(4):
                        for l in range(2):
                            img1 = f'/mnt/localssd/avadhoot/image_folders1/{fol}/1_0.png'
                            img2 = f'/mnt/localssd/avadhoot/results/{fol}/{d}_img_{i}_cond_{j}:{k}_{l}.png'
                            if(m == "fid"):
                                a1.append(fid_score(img1, img2))
                            if(m == "lpips"):
                                a1.append(lpips_score(img1, img2))
                            if(m == "ssim"):
                                a1.append(ssim_score(img1, img2))
                    res[f'{i}_{j}'] = sum(a1)/len(a1)   
            with open(f'/mnt/localssd/avadhoot/results/{fol}/{m}_{d}.json', 'w') as file:
                json.dump(res, file)