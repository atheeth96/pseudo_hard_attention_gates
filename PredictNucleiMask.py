import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import img_as_uint
from tqdm import tqdm
import os 
import numpy as np
import math
from skimage import img_as_ubyte
from skimage.transform import resize

import re


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import watershed,remove_small_holes,remove_small_objects,closing,area_closing
import scipy.ndimage as ndimage
from scipy import ndimage as ndi
from scipy.ndimage.morphology import (
                                    binary_erosion,
                                    binary_dilation, 
                                    binary_fill_holes,
                                    distance_transform_cdt,
                                    distance_transform_edt)
from skimage.feature import peak_local_max

import scipy.io



# def whole_dice_metric(y_pred,y_true):
#     smooth = 10e-16
#     # single image so just roll it out into a 1D array
    
#     m1 =np.reshape(y_pred,(-1))/255
#     m2 =np.reshape(y_true,(-1))/255
    
    
#     intersection = (m1 * m2)

#     score = 2. * (np.sum(intersection) + smooth) / (np.sum(m1) +(np.sum(m2) + smooth))
        
#     return score



def whole_img_pred(image1_path,image2_path,pred_dir_name,model,input_img1_ch=3,input_img2_ch=1,predict_boundary=False,patch_size=256,print_prompt=False):
    
    pred_dir=os.path.join(os.getcwd(),pred_dir_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.eval()
    
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
        if print_prompt:
            print("Made {} directory".format(pred_dir.split('/')[-1]))
    else:
        if print_prompt:
            print("{} directory already exists in {}".format(pred_dir.split('/')[-1],'/'.join(pred_dir.split('/')[:-1])))
        
    step_size=patch_size


    image1=imread(image1_path)
    img_name=image1_path.split('/')[-1]
    image2=imread(image2_path)
    
#     h_image=imread(h_path)
    
    r,c=image1.shape[:2]#4663,3881

    new_r_count=(math.ceil((r-patch_size)/patch_size)+1)#5
    new_c_count=(math.ceil((c-patch_size)/patch_size)+1)#5


    pad_r1=((new_r_count-1)*patch_size-r+patch_size)//2 #200
    pad_r2=((new_r_count-1)*patch_size-r+patch_size)-pad_r1 #200
    pad_c1=((new_c_count-1)*patch_size-c+patch_size)//2 #0
    pad_c2=((new_c_count-1)*patch_size-c+patch_size)-pad_c1#0
    
    window_shape=(patch_size,patch_size)
    
    if input_img1_ch==1:
        image1=np.amax(image1)-image1

        image1_padded=np.pad(image1, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant',\
                                constant_values=0)
        image1_patches=skimage.util.view_as_windows(image1_padded,window_shape, step=step_size)
        image1_patches=image1_patches.reshape((-1,patch_size,patch_size))
        image1_patches=image1_patches.transpose((0,2,1))
        image1_patches=np.expand_dims(image1_patches,axis=1)/255
        
        image1_patch_max=np.amax(image1_patches,axis=(1,2,3)).reshape(-1)
        
        
    else:
        image1_padded=np.pad(image1, [(pad_r1,pad_r2),(pad_c1,pad_c2),(0,0)], 'constant',\
                            constant_values=0)
        image1_patches=skimage.util.view_as_windows(image1_padded, (*window_shape,3), step=step_size)
        image1_patches=image1_patches.reshape((-1,patch_size,patch_size,3))
        image1_patches=image1_patches.transpose((0,3,2,1))/255
        
    if input_img2_ch is not None:    
        if input_img2_ch==1:
            image2=np.amax(image2)-image2
            image2_padded=np.pad(image2, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant',\
                                constant_values=0)
            image2_patches=skimage.util.view_as_windows(image2_padded, window_shape, step=step_size)
            image2_patches=image2_patches.reshape((-1,patch_size,patch_size))
            image2_patches=image2_patches.transpose((0,2,1))
            image2_patches=np.expand_dims(image2_patches,axis=1)/255

            image2_patch_max=np.amax(image2_patches,axis=(1,2,3)).reshape(-1)


        else:
            image2_padded=np.pad(image2, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant',\
                                constant_values=0)
            image2_patches=skimage.util.view_as_windows(image2_padded, (*window_shape,3), step=step_size)
            image2_patches=image2_patches.reshape((-1,patch_size,patch_size,3))
            image2_patches=image2_patches.transpose((0,3,2,1))/255
        
    
        
        
    nuclei_temp=[]
    
    
    
    attn_maps_temp={'attn_map_1':[],'attn_map_2':[],'attn_map_3':[],'attn_map_4':[],}
    
    if predict_boundary:
        bound_temp=[]

    for i in range(new_r_count):

        temp_img1_patches=torch.from_numpy(image1_patches[i*new_r_count:(i+1)*new_r_count]).type(torch.FloatTensor).to(device)
        if input_img2_ch is not None:
            temp_img2_patches=torch.from_numpy(image2_patches[i*new_r_count:\
                                                              (i+1)*new_r_count]).type(torch.FloatTensor).to(device)
            pred,attn_maps=model(temp_img1_patches,temp_img2_patches)
            del temp_img1_patches,temp_img2_patches
            
        else:
            pred,attn_maps=model(temp_img1_patches)
            del temp_img1_patches
            
        pred=torch.sigmoid(pred)
            

        
    
        if predict_boundary:
            nuclei,bound=torch.chunk(pred,2,dim=1)
            nuclei,bound=nuclei.detach().cpu().numpy(),bound.detach().cpu().numpy()
        else:
            nuclei=pred.detach().cpu().numpy()
            
        del pred
       
        nuclei=np.squeeze(nuclei,axis=1).transpose((0,2,1))
        nuclei=np.concatenate(nuclei,axis=1)
        nuclei_temp.append(nuclei)
        
        if predict_boundary:
            bound=np.squeeze(bound,axis=1).transpose((0,2,1))
            bound=np.concatenate(bound,axis=1)
            bound_temp.append(bound)
            
        for x,_ in enumerate(attn_maps):
            attn_maps[x]=np.squeeze(attn_maps[x].detach().cpu().numpy(),axis=1).transpose((0,2,1))
            
            current_shape=list(attn_maps[x].shape)
            
            if x!=0:
                attn_maps[x]=resize(attn_maps[x],(*current_shape[:1],*final_shape[1:]))
               
            else:
                final_shape=list(attn_maps[x].shape)
                
            attn_maps[x]=np.concatenate(attn_maps[x],axis=1)
            attn_maps_temp['attn_map_{}'.format(x+1)].append(attn_maps[x])
    
    nuclei_temp=np.array(nuclei_temp)
    nuclei_temp=np.concatenate(nuclei_temp,axis=0)
    nuclei_temp=nuclei_temp[pad_r1:nuclei_temp.shape[0]-pad_r2,pad_c1:nuclei_temp.shape[1]-pad_c2]*255
    nuclei_temp=nuclei_temp.astype(np.uint8)

    imsave(pred_dir_name+'/nuclei_'+img_name,nuclei_temp)
    
    if predict_boundary:

        bound_temp=np.array(bound_temp)
        bound_temp=np.concatenate(bound_temp,axis=0)
        bound_temp=bound_temp[pad_r1:bound_temp.shape[0]-pad_r2,pad_c1:bound_temp.shape[1]-pad_c2]*255
        bound_temp=bound_temp.astype(np.uint8)
        imsave(pred_dir_name+'/bound_'+img_name,bound_temp)
        
    for l,temp_attn in enumerate(attn_maps_temp):
        attn_array=np.array(attn_maps_temp[temp_attn])
        attn_array=np.concatenate(attn_array,axis=0)
        attn_array=attn_array[pad_r1:attn_array.shape[0]-pad_r2,pad_c1:attn_array.shape[1]-pad_c2]*255
        attn_array=attn_array.astype(np.uint8)
        imsave(pred_dir_name+'/attn_{}'.format(l+1)+img_name,attn_array)
        

    
#         nuclei_thresh=threshold_otsu(nuclei_temp)
#         nucei_thresholded=nuclei_temp>nuclei_thresh


    
    if print_prompt:
        print('Done')





def multiple_erosion(img,iter_count=5,selem=morphology.selem.disk(1)):
    for j in (range(iter_count)):
        img=morphology.binary_erosion(img, selem=selem)
    return img

def multiple_dialte(img,iter_count=5,selem=morphology.selem.disk(1)):
    for j in (range(iter_count)):
        img=morphology.binary_dilation(img, selem=selem)
    return img

def coord2array(coord):
    x=[]
    y=[]
    for i in coord:
        x.append(i[0])
        y.append(i[1])
    return (x,y)
def sort_n_array(img):
    img_labels=label(img)
    img_regions=regionprops(img_labels)
    final_gt=np.zeros_like(img)
    for i,region in enumerate(img_regions):
        coordinates=coord2array(list(region.coords))
        final_gt[coordinates]=i+1
        
    return final_gt



def watershed_seg(nuclei,boundary):
    
    def gen_inst_dst_map(nuclei):  
        shape = nuclei.shape[:2] # HW
        labeled_img=label(nuclei)
        labeled_img = remove_small_objects(labeled_img, min_size=50)
        regions=regionprops(labeled_img)
    

        canvas = np.zeros(shape, dtype=np.uint8)
        for region in regions:
            coordinates=coord2array(list(region.coords))
            nuc_map=np.zeros(shape)
            nuc_map[coordinates]=1  
            nuc_map=morphology.binary_dilation(nuc_map, selem=morphology.selem.disk(2)).astype(np.uint8)
            nuc_dst = ndi.distance_transform_edt(nuc_map)
            nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))       
            canvas += nuc_dst.astype('uint8')
        return canvas
    
    nuclei=nuclei>0.45*255#threshold_otsu(nuclei)
    
    
    nuclei=nuclei.astype(np.uint8)
    
    
    nuclei=area_closing(nuclei,20)
    nuclei=closing(nuclei.astype(np.uint8),morphology.selem.square(2))
    nuclei = binary_fill_holes(nuclei)

    nuclei=ndimage.binary_fill_holes(nuclei).astype(int)
    

    
    boundary=boundary>0.3*255#120
    boundary=boundary.astype(np.uint8)
    
    
    nuclei_seeds_ini=nuclei-boundary
    nuclei_seeds_ini[np.where(nuclei_seeds_ini<=0)]=0
    nuclei_seeds_ini[np.where(nuclei_seeds_ini>0)]=1
    
    nuclei_seeds=morphology.binary_erosion(nuclei_seeds_ini, selem=morphology.selem.disk(2)).astype(np.uint8)
    

    labeled_img=label(nuclei_seeds)
    labeled_img = remove_small_objects(labeled_img, min_size=50)
    
    regions=regionprops(labeled_img)

    final_image=np.zeros_like(nuclei_seeds_ini)
    distance = gen_inst_dst_map(nuclei_seeds_ini)
    markers = ndi.label(nuclei_seeds)[0]
    final_image = watershed(-distance, markers, mask=nuclei,watershed_line=False)
    return final_image


def retrive_gt(path,img_name):
    if 'CPM_17' in path:
        f = open(path+'_'.join(img_name.split('_')[1:]).split('.')[0]+'_mask.txt', 'r')
        x = f.readlines()

        gt=np.array([int(a) for a in x[1:]]).reshape(list(map(int, re.findall('\d+', x[0]))))
        return gt
    elif 'CoNSeP' in path:
        gt=np.load(path+'_'.join(img_name.split('_')[1:]).split('.')[0]+'.npy')[:,:,0]
        return gt
        
    elif 'kumar' in path:
        return imread(path+img_name.split('.')[0].split('_')[1:][0]+'.tif')
        
    else:
        print("Wrong path")