import numpy as np
import os
import imutils

import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import draw
from skimage.io import imread,imsave
from skimage.filters import threshold_otsu
from skimage.util import pad
import skimage
import time

import math
from DatasetCreator import create_data_vahadane
from tqdm import tqdm 
from PatchExtractor import PatchExtractor
import warnings
warnings.filterwarnings('ignore')

'''
This script converts the xml files to 2 binary images (one for boundary maps and another for nuclei maps)

'''

png_dir='png_images'                              #The directory with converted tif images in png format
annotation_dir='Annotations'                      #The directory with Annotation files
nucleus_dir='processed_data/nucleus_maps'         #The directory (need not be created) to store nuclei maps
boundary_dir='processed_data/boundary_maps'       #The directory (need not be created) to store boundary maps

no_annotation_files=len([x for x in os.listdir(annotation_dir) if x.split('.')[-1].lower()=='xml'])
no_png_files=len([x for x in os.listdir(png_dir) if x.split('.')[-1].lower()=='png'])
no_nuclei_masks=len([x for x in os.listdir(nucleus_dir) if x.split('.')[-1].lower()=='png'])
no_boundary_masks=len([x for x in os.listdir(boundary_dir) if x.split('.')[-1].lower()=='png'])
if no_annotation_files!=no_nuclei_masks or no_annotation_files!=no_boundary_masks:
    create_data_vahadane(png_dir,annotation_dir,nucleus_dir,boundary_dir)
    
else:
    print("Binary maps are already present")
    
    
# The train and test stratification at a slide level. The split needs to be near 50% to better estimte generalization ability   
train_list=['TCGA-A7-A13E-01Z-00-DX1.png',\
'TCGA-A7-A13F-01Z-00-DX1.png',\
'TCGA-AR-A1AK-01Z-00-DX1.png',\
'TCGA-AR-A1AS-01Z-00-DX1.png',\
'TCGA-18-5592-01Z-00-DX1.png',\
'TCGA-38-6178-01Z-00-DX1.png',\
'TCGA-49-4488-01Z-00-DX1.png',\
'TCGA-50-5931-01Z-00-DX1.png',\
'TCGA-HE-7130-01Z-00-DX1.png',\
'TCGA-HE-7129-01Z-00-DX1.png',\
'TCGA-B0-5711-01Z-00-DX1.png',\
'TCGA-B0-5698-01Z-00-DX1.png',\
'TCGA-G9-6362-01Z-00-DX1.png',\
'TCGA-G9-6336-01Z-00-DX1.png',\
'TCGA-G9-6363-01Z-00-DX1.png',\
'TCGA-G9-6356-01Z-00-DX1.png']

test_list=list(set([x for x in os.listdir(png_dir) if x.split('.')[-1].lower()=='png'])-set(train_list))


# The slide images are first segregated into seperate train and test directory. So are the associated grund truths

h_e_train_dir='processed_data/h_e_train_dir'                        # H&E dir to store training images 
h_train_dir='processed_data/h_train_dir'                            # H_gray_maps dir to store training images 
nuclei_mask_train_dir='processed_data/nuclei_mask_train_dir'        # nuclei_maps dir to store training images 
boundary_mask_train_dir='processed_data/boundary_mask_train_dir'    # boundary_maps dir to store training images 
if not os.path.exists(h_e_train_dir):
    os.mkdir(h_e_train_dir)
if not os.path.exists(h_train_dir):
    os.mkdir(h_train_dir)
if not os.path.exists(nuclei_mask_train_dir):
    os.mkdir(nuclei_mask_train_dir)
if not os.path.exists(boundary_mask_train_dir):
    os.mkdir(boundary_mask_train_dir)
    
for img in tqdm(train_list):
    nuclei=img.split('.')[0]+'_nucleus_mask.png'
    boundary=img.split('.')[0]+'_boundary_mask.png'
    os.system("cp {} {}".format(os.path.join('norm_ideal',img),os.path.join(h_e_train_dir,img)))
    os.system("cp {} {}".format(os.path.join('H_gray',img),os.path.join(h_train_dir,img)))
    os.system("cp {} {}".format(os.path.join('processed_data/nucleus_maps',nuclei),os.path.join(nuclei_mask_train_dir,nuclei)))
    os.system("cp {} {}".format(os.path.join('processed_data/boundary_maps',boundary)\
                                ,os.path.join(boundary_mask_train_dir,boundary)))


    
    
h_e_test_dir='processed_data/h_e_test_dir'                        # H&E dir to store testing images 
h_test_dir='processed_data/h_test_dir'                            # H_gray_maps dir to store testing images 
nuclei_mask_test_dir='processed_data/nuclei_mask_test_dir'        # nuclei_maps dir to store testing images 
boundary_mask_test_dir='processed_data/boundary_mask_test_dir'    # boundary_maps dir to store testing images 
if not os.path.exists(h_e_test_dir):
    os.mkdir(h_e_test_dir)
if not os.path.exists(h_test_dir):
    os.mkdir(h_test_dir)
if not os.path.exists(nuclei_mask_test_dir):
    os.mkdir(nuclei_mask_test_dir)
if not os.path.exists(boundary_mask_test_dir):
    os.mkdir(boundary_mask_test_dir)
    
    
for img in tqdm(test_list):
    nuclei=img.split('.')[0]+'_nucleus_mask.png'
    boundary=img.split('.')[0]+'_boundary_mask.png'
    os.system("cp {} {}".format(os.path.join('norm_ideal',img),os.path.join(h_e_test_dir,img)))
    os.system("cp {} {}".format(os.path.join('H_gray',img),os.path.join(h_test_dir,img)))
    os.system("cp {} {}".format(os.path.join('processed_data/nucleus_maps',nuclei),os.path.join(nuclei_mask_test_dir,nuclei)))
    os.system("cp {} {}".format(os.path.join('processed_data/boundary_maps',boundary)\
                                ,os.path.join(boundary_mask_test_dir,boundary)))
        

    
# Once the slides are segregated, the patches are generated offline

# TRAIN PATCHES
h_e_train_patch_dir='processed_data/h_e_train_patch_dir'                           # The directory for H&E train patches         
h_train_patch_dir='processed_data/h_train_patch_dir'                               # The directory for H train patches 
nuclei_mask_train_patch_dir='processed_data/nuclei_mask_train_patch_dir'           # The directory for nuclei map train patches 
boundary_mask_train_patch_dir='processed_data/boundary_mask_train_patch_dir'       # The directory for boundary map train patches 

if not all([os.path.exists(h_e_train_patch_dir),\
os.path.exists(h_train_patch_dir),\
os.path.exists(nuclei_mask_train_patch_dir),\
os.path.exists(boundary_mask_train_patch_dir)]):
    try:
        os.system("rm -rf {}".format(h_e_train_patch_dir))
        os.system("rm -rf {}".format(h_train_patch_dir))
        os.system("rm -rf {}".format(nuclei_mask_train_patch_dir))
        os.system("rm -rf {}".format(boundary_mask_train_patch_dir))
    except:
        print('Redundant patches maybe present. Recommend deleting entire dir and trying again')
        

    train_extractor=PatchExtractor(h_e_train_dir,h_train_dir,nuclei_mask_train_dir\
                     ,boundary_mask_train_dir,h_e_train_patch_dir,h_train_patch_dir,\
                     nuclei_mask_train_patch_dir,boundary_mask_train_patch_dir,gray=True,progress_bar=True)
    train_extractor.extract_patches(patch_size=256,step=128 )
    
else:
    print("TRAIN PATCHES ALREADY PRESENT")

# TEST PATCHES
h_e_test_patch_dir='processed_data/h_e_test_patch_dir'                               # The directory for H&E test patches
h_test_patch_dir='processed_data/h_test_patch_dir'                                   # The directory for H test patches 
nuclei_mask_test_patch_dir='processed_data/nuclei_mask_test_patch_dir'               # The directory for nuclei map test patches
boundary_mask_test_patch_dir='processed_data/boundary_mask_test_patch_dir'           # The directory for bound map test patches
if not all([os.path.exists(h_e_test_patch_dir),\
os.path.exists(h_test_patch_dir),\
os.path.exists(nuclei_mask_test_patch_dir),\
os.path.exists(boundary_mask_test_patch_dir)]):
    try:
        os.system("rm -rf {}".format(h_e_test_patch_dir))
        os.system("rm -rf {}".format(h_test_patch_dir))
        os.system("rm -rf {}".format(nuclei_mask_test_patch_dir))
        os.system("rm -rf {}".format(boundary_mask_test_patch_dir))
    except:
        print('Redundant patches maybe present. Recommend deleting entire dir and trying again')
    


    test_extractor=PatchExtractor(h_e_test_dir,h_test_dir,nuclei_mask_test_dir\
                     ,boundary_mask_test_dir,h_e_test_patch_dir,h_test_patch_dir,\
                     nuclei_mask_test_patch_dir,boundary_mask_test_patch_dir,gray=True,progress_bar=True)
    test_extractor.extract_patches(patch_size=256,step=128)
    
    
    
else:
    print("TEST PATCHES ALREADY PRESENT")