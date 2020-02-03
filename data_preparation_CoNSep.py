import os
from tqdm import tqdm
import skimage
from skimage.io import imread,imsave
import numpy as np
from skimage import morphology

import warnings
warnings.filterwarnings('ignore')

import math
'''
    Python script to prepare binary maps of nuclei and boundary from CoNSep data's numpy arrays'
    
'''

patch_size=256
step=128
gray=True
for tag in ['Train','Test']:
    path_images='other_data/CoNSeP/{}/Images'.format(tag)
    path_h_gray='other_data/CoNSeP/{}/H_gray'.format(tag)
    path_bound='other_data/CoNSeP/{}/BoundaryMaps'.format(tag)
    path_gt='other_data/CoNSeP/{}/Labels'.format(tag)
    path_nuclei='other_data/CoNSeP/{}/NucleiMaps'.format(tag)
    if not all([len(os.listdir(path_bound))==len(os.listdir(path_nuclei)),\
                len(os.listdir(path_nuclei))==len(os.listdir(path_gt))]):
        
        if not os.path.exists(path_bound):
            os.mkdir(path_bound)

        if not os.path.exists(path_nuclei):
            os.mkdir(path_nuclei)

        img_list=[x for x in os.listdir() if x.split('.')[-1]=='npy']


        for img_name in tqdm(img_list):

            img=np.load('other_data/CoNSeP/{}/Labels/{}'.format(tag,img_name))[:,:,0]

            boundary_map=np.zeros(img.shape[:2],dtype=np.uint8)

            for i in range(1,int(np.amax(img))):
                nuclei_map=np.zeros(img.shape,dtype=np.uint8)
                nuclei_map[np.where(img==i)]=255

                contours=skimage.measure.find_contours(nuclei_map,0)[0]

                img_temp=np.zeros(img.shape,dtype=np.uint8)
                for countour in contours:
                    x,y=countour

                    img_temp[int(x),int(y)]=255

                boundary_map+=img_temp

            boundary_map=morphology.dilation(boundary_map, selem=None)
            boundary_map=boundary_map.astype(np.uint8)
            imsave(path_bound+'/'+img_name.split('.')[0]+'.png',boundary_map)

            img[np.where(img!=0)]=255
            img=img.astype(np.uint8)
            imsave(path_nuclei+'/'+img_name.split('.')[0]+'.png',img)
            
    input1_train_patch_dir='other_data/CoNSeP/{}/H_E_patches'.format(tag)
    input2_train_patch_dir='other_data/CoNSeP/{}/H_patches'.format(tag)
 
    nuclei_train_patch_dir='other_data/CoNSeP/{}/nuclei_patches'.format(tag)
    boundary_train_patch_dir='other_data/CoNSeP/{}/boundary_patches'.format(tag)
    
    for directory in [input1_train_patch_dir,input2_train_patch_dir,nuclei_train_patch_dir,boundary_train_patch_dir]:
        if os.path.exists(directory):
            os.system("rm -rf {}".format(directory))
    
    for directory in [input1_train_patch_dir,input2_train_patch_dir,nuclei_train_patch_dir,boundary_train_patch_dir]:
        os.mkdir(directory)

    loop=tqdm([x for x in os.listdir(path_images) if x.split('.')[-1]=='png'])
    
    exception_list=[]
    NO_PATCHES=0
    for img_name in loop:

        try:
            h_e_img=imread(os.path.join(path_images,img_name))
            h_img=imread(os.path.join(path_h_gray,img_name))

            nuclei_mask=imread(os.path.join(path_nuclei,img_name))
            boundary_mask=imread(os.path.join(path_bound,img_name))


            loop.set_postfix(Image=img_name)

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


            if gray:
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

                imsave(os.path.join(input1_train_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1)),h_e_patch)
                imsave(os.path.join(input2_train_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1)),h_patches[i])

                imsave(os.path.join(nuclei_train_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1))\
                       ,nuclei_patches[i])
                imsave(os.path.join(boundary_train_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1))\
                       ,boundary_patches[i])


        except:
            exception_list.append(img_name.split('.')[0])
    print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')





