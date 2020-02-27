import argparse
import logging
import os

import numpy as np
from PIL import Image

from pathlib import Path
import glob

def dice_score(result_folder, gth_folder, gth_ext='gif'):
    '''
    Note result_folder contains the segmentation results.
    Files in result_folder should be names like cremi_00000.jpg
    Files in gth_folder should be in .gif form and named like cremi_00000_mask.gif
    '''
    dice_list = []
    os.chdir(result_folder)
    for file in glob.glob("*"):
        res_name = os.path.join(result_folder, file)
        gth_name = os.path.join(gth_folder, file[:-4]+'_mask.'+gth_ext)
        res = Image.open(res_name)
        res = np.array(res)
        res[res < 127.5] = 0
        res[res >= 127.5] = 1
        gth = Image.open(gth_name)
        gth = np.array(gth)
        assert(res.shape == gth.shape)
        
        #eps = 0.0001
        #res_flatten = res.flatten()
        #gth_flatten = gth.flatten()
        #inter = np.dot(res_flatten, gth_flatten)
        #union = np.sum(res_flatten) + np.sum(gth_flatten) + eps
        #t = (2 * inter + eps) / union
        
        t = np.sum(res[gth==1])*2.0 / (np.sum(res) + np.sum(gth))
        
        dice_list.append(t)
    return np.mean(dice_list)

def get_args():
    parser = argparse.ArgumentParser(description='Compute pixel accuracy',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.add_argument('--resfolder', type=str, default='./',
                        help="Folder where segmentation results are contained")
    parser.add_argument('--gthfolder', type=str, default='./',
                        help="Folder where ground truth results (.gif) are contained")
    parser.add_argument('--gthext', type=str, default='gif',
                        help="The extension of the ground truth files")


    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    pix_accu = dice_score(args.resfolder, args.gthfolder, args.gthext)
    print(pix_accu)