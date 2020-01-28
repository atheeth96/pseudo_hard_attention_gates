

import numpy as np
import glob
import os
import sys
import scipy

import matplotlib.pyplot as plt

import imutils
import re


import zipfile

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction import image


import skimage
from skimage.util import pad
from skimage import draw
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filters import threshold_otsu

import time
import math

import random
from tqdm import tqdm 

class PatchExtractor():
    def __init__(self,h_e_dir,h_dir,nuclei_mask_dir\
                 ,boundary_mask_dir,h_e_patch_dir,h_patch_dir,\
                 nuclei_mask_patch_dir,boundary_mask_patch_dir,gray=True,progress_bar=True):
        
        self.h_e_dir=h_e_dir
        self.h_dir=h_dir
        self.nuclei_mask_dir=nuclei_mask_dir
        self.boundary_mask_dir=boundary_mask_dir
        
        self.h_e_patch_dir=h_e_patch_dir
        self.h_patch_dir=h_patch_dir
        self.nuclei_mask_patch_dir=nuclei_mask_patch_dir
        self.boundary_mask_patch_dir=boundary_mask_patch_dir
        
        self.gray=gray
        self.progress_bar=progress_bar
        

    def extract_patches(self,patch_size=512,step=512):
        
        NO_PATCHES=0
       
        if not os.path.exists(self.h_e_patch_dir):
            os.mkdir(self.h_e_patch_dir)
            if self.progress_bar:
                print("H&E patch directory made")
        if not os.path.exists(self.h_patch_dir):
            os.mkdir(self.h_patch_dir)
            if self.progress_bar:
                print("H patch directory made")
        if not os.path.exists(self.nuclei_mask_patch_dir):
            os.mkdir(self.nuclei_mask_patch_dir)
            if self.progress_bar:
                print("nuclei mask patch directory made")
        if not os.path.exists(self.boundary_mask_patch_dir):
            os.mkdir(self.boundary_mask_patch_dir)
            if self.progress_bar:
                print("boundary mask patch directory made")
      
        exception_list=[]
        
       
        loop=tqdm([x for x in os.listdir(self.h_e_dir) if '.png' in x])
        
        for h_e_img_name in loop:
            
            try:
                h_e_img=imread(os.path.join(self.h_e_dir,h_e_img_name))
                h_img=imread(os.path.join(self.h_dir,h_e_img_name))

                nuclei_mask=imread(os.path.join(self.nuclei_mask_dir,h_e_img_name.split('.')[0]+'_nucleus_mask.png'))
                boundary_mask=imread(os.path.join(self.boundary_mask_dir,h_e_img_name.split('.')[0]+'_boundary_mask.png'))


                loop.set_postfix(Image=h_e_img_name)

                r,c=h_e_img.shape[:2]#1000,1000

                new_r_count=(math.ceil((r-patch_size)/step)+1)#15
                new_c_count=(math.ceil((c-patch_size)/step)+1)#15



                pad_r1=((new_r_count-1)*step-r+patch_size)//2 #228
                pad_r2=((new_r_count-1)*step-r+patch_size)-pad_r1 #229
                pad_c1=((new_c_count-1)*step-c+patch_size)//2 #107
                pad_c2=((new_c_count-1)*step-c+patch_size)-pad_c1#108


                h_e_img_padded=np.pad(h_e_img,[(pad_r1,pad_r2),(pad_c1,pad_c2),(0,0)],\
                                      'constant',constant_values=0)                    



                nuclei_mask_padded=np.pad(nuclei_mask, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
                boundary_mask_padded=np.pad(boundary_mask, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)

                window_shape=(patch_size,patch_size,3) 
                window_shape_mask=(patch_size,patch_size) 

                h_e_patches=skimage.util.view_as_windows(h_e_img_padded, window_shape, step=step)
                h_e_patches=h_e_patches.reshape((-1,patch_size,patch_size,3))


                if self.gray:
                    h_img_padded=np.pad(h_img, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
                    h_patches=skimage.util.view_as_windows(h_img_padded, window_shape_mask, step=step)
                    h_patches=h_patches.reshape((-1,patch_size,patch_size))
                else:
                    h_img_padded=np.pad(h_img, [(pad_r1,pad_r2),(pad_c1,pad_c2),(0,0)], 'constant', constant_values=0)
                    h_patches=skimage.util.view_as_windows(h_img_padded, window_shape, step=step)
                    h_patches=h_patches.reshape((-1,patch_size,patch_size,3))



                nuclei_patches=skimage.util.view_as_windows(nuclei_mask_padded, window_shape_mask, step=step)
                nuclei_patches=nuclei_patches.reshape((-1,patch_size,patch_size))

                boundary_patches=skimage.util.view_as_windows(boundary_mask_padded, window_shape_mask, step=step)
                boundary_patches=boundary_patches.reshape((-1,patch_size,patch_size))




                for i,h_e_patch in enumerate(h_e_patches):
                    NO_PATCHES+=1

                    imsave(os.path.join(self.h_e_patch_dir,h_e_img_name.split('.')[0]+'_{}_.png'.format(i+1)),h_e_patch)
                    imsave(os.path.join(self.h_patch_dir,h_e_img_name.split('.')[0]+'_{}_.png'.format(i+1)),h_patches[i])

                    imsave(os.path.join(self.nuclei_mask_patch_dir,h_e_img_name.split('.')[0]+'_{}_.png'.format(i+1))\
                           ,nuclei_patches[i])
                    imsave(os.path.join(self.boundary_mask_patch_dir,h_e_img_name.split('.')[0]+'_{}_.png'.format(i+1))\
                           ,boundary_patches[i])
                    

            except:
                exception_list.append(h_e_img_name.split('.')[0])
        if self.progress_bar:
            print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')