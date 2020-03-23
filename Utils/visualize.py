import numpy as np
import skimage
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

def blend_img(img1,img2):
    img1=cv2.applyColorMap(img1, 2)
    return cv2.addWeighted(img1, 0.68, img2, 0.32, 0)

for mode in ['attn_unet','DEAU']:

    for dataset in ['CoNSeP','CPM_17','kumar']:
        maps_dir='/datalab/training-assets/R_medical/atheeth/nuclei_seg/Results/saliency_maps_{}_{}'.format(mode,dataset)
        if not os.path.exists(maps_dir):
            os.mkdir(maps_dir)
        pred_dir='/datalab/training-assets/R_medical/atheeth/nuclei_seg/Results/prediction_{}_{}'.format(mode,dataset)
        img_list=['_'.join(x.split('_')[1:])[1:] for x in os.listdir(pred_dir) if 'attn_1' in x]
        for img in tqdm(img_list):
            attn_1=imread(os.path.join(pred_dir,'attn_1{}'.format(img)))
            attn_2=imread(os.path.join(pred_dir,'attn_2{}'.format(img)))
            attn_3=imread(os.path.join(pred_dir,'attn_3{}'.format(img)))
            attn_4=imread(os.path.join(pred_dir,'attn_4{}'.format(img)))
            if dataset=='CPM_17':
                ip_img=imread('/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CPM_17/Test/Images/{}'.format(img))[:,:,:3]
            elif dataset=='kumar':
                ip_img=imread('/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/kumar/processed_data/h_e_test_dir/{}'.format(img))

            else:
                ip_img=imread('/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CoNSeP/Test/Images/{}'.format(img))[:,:,:3]
                
                
            for i,attn_map in enumerate([attn_1,attn_2,attn_3,attn_4]):
                blended_img=blend_img(attn_map,ip_img)
                imsave(maps_dir+'/attn_{}_{}_'.format(i+1,mode)+img,blended_img)