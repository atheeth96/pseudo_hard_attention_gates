
import os
from tqdm import tqdm
import skimage
from skimage.io import imread,imsave
import numpy as np
from skimage import morphology
import numpy


from skimage.measure import label, regionprops

import re
import warnings
warnings.filterwarnings('ignore')

import math
'''
    Python script to prepare binary maps of nuclei and boundary from CPM_17 data's numpy arrays'
    
'''
def coord2array(coord):
    x=[]
    y=[]
    for i in coord:
        x.append(i[0])
        y.append(i[1])
    return (x,y)

patch_size=256
step=128
gray=True
for tag in ['Train','Test']:
    path_images='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/Images'.format(tag)
    path_h_gray='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/H_gray'.format(tag)
    path_bound='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/BoundaryMaps'.format(tag)
    
    path_gt='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/GT_Mask'.format(tag)
    path_nuclei='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/NucleiMaps'.format(tag)
    path_verical='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/vertical_maps'.format(tag)
    path_horizontal='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/horizontal_maps'.format(tag)
    
    if not all([os.path.exists(path_bound),os.path.exists(path_nuclei)]):
        
        if not os.path.exists(path_bound):
            os.mkdir(path_bound)

        if not os.path.exists(path_nuclei):
            os.mkdir(path_nuclei)
            
        if not os.path.exists(path_verical):
            os.mkdir(path_verical)
        
        if not os.path.exists(path_horizontal):
            os.mkdir(path_horizontal)
            

        img_list=[x for x in os.listdir(path_gt) if x.split('.')[-1]=='txt']


        for img_name in tqdm(img_list):
            
            f = open(path_gt+'/'+img_name, 'r')
            x = f.readlines()
            img=np.array([int(a) for a in x[1:]]).reshape(list(map(int, re.findall('\d+', x[0]))))

            boundary_map=np.zeros(img.shape[:2],dtype=np.uint8)
            
            label_img = label(img)
            regions = regionprops(label_img)
            
            
            ############
            x_temp_img=np.zeros_like(img,dtype=np.float32)
            y_temp_img=np.zeros_like(img,dtype=np.float32)
            for i,region in enumerate(regions):
                
                coordinates=coord2array(list(region.coords))
                
                nuclei_map=np.zeros(img.shape,dtype=np.uint8)
                nuclei_map[coordinates]=255
                
                contours=skimage.measure.find_contours(nuclei_map,0)[0]
                
                img_temp=np.zeros(img.shape,dtype=np.uint8)
                for countour in contours:
                    x,y=countour

                    img_temp[int(x),int(y)]=255
                    
                boundary_map+=img_temp
                
                distance_x=(coordinates[0]-np.ones(len(coordinates[0]))*round(region.centroid[0]))
                distance_y=(coordinates[1]-np.ones(len(coordinates[1]))*round(region.centroid[1]))
                distance_x=((distance_x-np.min(distance_x))/(np.max(distance_x)-np.min(distance_x)))*(2) -1
                distance_y=((distance_y-np.min(distance_y))/(np.max(distance_y)-np.min(distance_y)))*(2) -1
                x_temp_img[coordinates]=distance_x
                y_temp_img[coordinates]=distance_y
            ############

#             for i in range(1,int(np.amax(img))):
#                 nuclei_map=np.zeros(img.shape,dtype=np.uint8)
#                 nuclei_map[np.where(img==i)]=255

#                 contours=skimage.measure.find_contours(nuclei_map,0)[0]

#                 img_temp=np.zeros(img.shape,dtype=np.uint8)
#                 for countour in contours:
#                     x,y=countour

#                     img_temp[int(x),int(y)]=255

#                 boundary_map+=img_temp

            boundary_map=morphology.dilation(boundary_map, selem=None)
            boundary_map=boundary_map.astype(np.uint8)
            imsave(path_bound+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1])+'.png',boundary_map.astype(np.uint8))

            img[np.where(img!=0)]=255
            img=img.astype(np.uint8)
            imsave(path_nuclei+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1])+'.png',img.astype(np.uint8))
            
#             imsave(path_verical+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1])+'.png',x_temp_img.astype(np.uint8))
#             imsave(path_horizontal+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1])+'.png',y_temp_img.astype(np.uint8))
            np.save(path_horizontal+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1]),y_temp_img)
            np.save(path_verical+'/'+'_'.join(img_name.split('.')[0].split('_')[:-1]),x_temp_img)
            
            
    input1_train_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/H_E_patches'.format(tag)
    input2_train_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/H_patches'.format(tag)
 
    nuclei_train_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/nuclei_patches'.format(tag)
    boundary_train_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/boundary_patches'.format(tag)
    
    vertical_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/vertical_patches'.format(tag)
    horizontal_patch_dir='/'.join(os.getcwd().split('/')[:-1])+'/Data/CPM_17/{}/horizontal_patches'.format(tag)
    
    for directory in [input1_train_patch_dir,input2_train_patch_dir,nuclei_train_patch_dir\
                      ,boundary_train_patch_dir,vertical_patch_dir,horizontal_patch_dir]:
        if os.path.exists(directory):
            os.system("rm -rf {}".format(directory))
    
    for directory in [input1_train_patch_dir,input2_train_patch_dir,nuclei_train_patch_dir,\
                      boundary_train_patch_dir,vertical_patch_dir,horizontal_patch_dir]:
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

            vertical_map=np.load(os.path.join(path_verical,img_name.split('.')[0]+'.npy'))
            horizontal_map=np.load(os.path.join(path_horizontal,img_name.split('.')[0]+'.npy'))


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

            vertical_map_padded=np.pad(vertical_map, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
            horizontal_map_padded=np.pad(horizontal_map, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)

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

            vertical_map_patches=skimage.util.view_as_windows(vertical_map_padded, window_shape_mask, step=step)
            vertical_map_patches=vertical_map_patches.reshape((-1,patch_size,patch_size))
            horizontal_map_patches=skimage.util.view_as_windows(horizontal_map_padded, window_shape_mask, step=step)
            horizontal_map_patches=horizontal_map_patches.reshape((-1,patch_size,patch_size))






            for i,h_e_patch in enumerate(h_e_patches):
                NO_PATCHES+=1

                imsave(os.path.join(input1_train_patch_dir,img_name.split('.')[0]+'_{}.png'.format(i+1)),h_e_patch)
                imsave(os.path.join(input2_train_patch_dir,img_name.split('.')[0]+'_{}.png'.format(i+1)),h_patches[i])

                imsave(os.path.join(nuclei_train_patch_dir,img_name.split('.')[0]+'_{}.png'.format(i+1))\
                       ,nuclei_patches[i])
                imsave(os.path.join(boundary_train_patch_dir,img_name.split('.')[0]+'_{}.png'.format(i+1))\
                       ,boundary_patches[i])
#                 imsave(os.path.join(vertical_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1))\
#                        ,vertical_map_patches[i])
#                 imsave(os.path.join(horizontal_patch_dir,img_name.split('.')[0]+'_{}_.png'.format(i+1))\
#                        ,horizontal_map_patches[i])
                np.save(os.path.join(vertical_patch_dir,img_name.split('.')[0]+'_{}'.format(i+1)),vertical_map_patches[i])
                np.save(os.path.join(horizontal_patch_dir,img_name.split('.')[0]+'_{}'.format(i+1)),horizontal_map_patches[i])
                


        except:
            exception_list.append(img_name.split('.')[0])
    print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')





