# Loss utils used by denoise_dir.py
# TODO use PIQA

import torch
import torchvision
import os
from PIL import Image
from libs import pytorch_ssim
from dataset_torch_3 import sort_ISOs
import argparse

totensor = torchvision.transforms.ToTensor()

def find_gt_path(denoised_fn, gt_dir):
    dsname, setdir = denoised_fn.split('_')[0:2]
    setfiles = os.listdir(os.path.join(gt_dir, setdir))
    ext = setfiles[0].split('.')[-1]
    isos = [fn.split('_')[2][:-4] for fn in setfiles]
    baseiso = sort_ISOs(isos)[0][0]
    baseiso_fn = dsname+'_'+setdir+'_'+baseiso+'.'+ext
    gt_fpath = os.path.join(gt_dir, setdir, baseiso_fn)
    return gt_fpath

def files(path):
    for fn in os.listdir(path):
        if os.path.isfile(os.path.join(path, fn)) and fn != 'res.txt':
            yield fn

def gen_score(noisy_dir, gt_dir='../../datasets/test/NIND/ds_fs', device=torch.device('cuda:0')):
    MSE = torch.nn.MSELoss().to(device)
    SSIM = pytorch_ssim.SSIM().to(device)
    with open(os.path.join(noisy_dir, 'res.txt'), 'w') as f:
        for noisy_img in files(noisy_dir):
            gtpath = find_gt_path(noisy_img, gt_dir)
            noisy_path = os.path.join(noisy_dir, noisy_img)
            gtimg = totensor(Image.open(gtpath).convert('RGB')).to(device)
            noisyimg = totensor(Image.open(noisy_path).convert('RGB')).to(device)
            gtimg = gtimg.reshape([1]+list(gtimg.shape))
            noisyimg = noisyimg.reshape([1]+list(noisyimg.shape))
            MSELoss = MSE(gtimg, noisyimg).item()
            SSIMScore = SSIM(gtimg, noisyimg).item()
            res =noisy_img+','+str(SSIMScore)+','+str(MSELoss)
            print(res)
            f.write(res+'\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get SSIM score and MSE loss from test images')
    parser.add_argument('--noisy_dir', type=str, required=True, help="Noisy / denoised data directory")
    parser.add_argument('--gt_dir', type=str, default='../../datasets/test/NIND/ds_fs', help='Ground truths directory')
    args, _ = parser.parse_known_args()
    gen_score(args.noisy_dir, args.gt_dir)
