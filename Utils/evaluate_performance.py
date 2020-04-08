import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg')

from PredictNucleiMask import whole_img_pred,watershed_seg,retrive_gt
from Metrics import get_fast_aji,get_fast_aji_plus,get_fast_pq,get_fast_dice_2,remap_label,whole_dice_metric

import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from skimage.io import imread,imsave
from skimage.filters import threshold_otsu

import imageio.core.util

# def ignore_warnings(*args, **kwargs):
#     pass

# imageio.core.util._precision_warn = ignore_warnings


pred_dir='/datalab/training-assets/R_medical/atheeth/nuclei_seg/Results/prediction_ab_dual_h_CPM_17'
gt_dir='/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CPM_17/Test/GT_Mask/'

df =pd.DataFrame(columns=['Image','Dice','AJI','DQ','SQ','PQ'])
img_list=[x for x in os.listdir(pred_dir) if 'nuclei_' in x and 'proc' not in x and 'GT' not in x]
avg_aji=0
avg=0
dq_avg=0
sq_avg=0
pq_avg=0
loop=tqdm(img_list)
for img_name in loop:

    nuclei=imread(pred_dir+'/'+img_name)    
    boundary=imread((pred_dir+'/'+'bound_'+'_'.join(img_name.split('_')[1:])))

    nuclei_temp=np.expand_dims(nuclei,axis=2)
    boundary_temp=np.expand_dims(boundary,axis=2)

    pred=np.concatenate((nuclei_temp,boundary_temp),axis=2)

    gt=retrive_gt(gt_dir,img_name)
    gt=remap_label(gt)
    dice_gt=(gt>0).astype(np.uint8)*255
    
    img_pred=watershed_seg(nuclei,boundary)
    img_pred=remap_label(img_pred)
    dice_img=(img_pred>0).astype(np.uint8)*255

    imsave(pred_dir+'/'+img_name.split('.')[0]+'_GT.tif',gt.astype(np.int16),check_contrast=False)
    imsave(pred_dir+'/'+img_name.split('.')[0]+'_proc.tif',img_pred.astype(np.int16),check_contrast=False)
    
    img_pred=img_pred.astype(np.int16)
    
    dice=whole_dice_metric(dice_img,dice_gt)
    aji=get_fast_aji(gt,img_pred)
    pq,_=get_fast_pq(gt,img_pred)
    dq=pq[0]
    sq=pq[1]
    pq=pq[-1]
    df = df.append({'Image': img_name.split('.')[0],'Dice': dice,\
                        'AJI': aji,'DQ':dq,'SQ':sq,'PQ':pq}, ignore_index=True)
    
    loop.set_postfix(dice=dice,aji=aji,dq=dq,sq=sq,pq=pq)

    avg+=dice
    avg_aji+=aji
    dq_avg+=dq
    sq_avg+=sq
    pq_avg+=pq
    
    
print("DICE : ",avg/len(img_list))
print("AJI : ",avg_aji/len(img_list))
print("DQ : ",dq_avg/len(img_list))
print("SQ : ",sq_avg/len(img_list))
print("PQ : ",pq_avg/len(img_list))
df.to_csv(os.path.join(pred_dir,'predictions.csv'),index=False)