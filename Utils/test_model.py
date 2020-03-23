import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg')

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.io import imread,imsave
from skimage.filters import threshold_otsu
import skimage
import warnings
warnings.filterwarnings('ignore')
# from Models_v1 import DualEncoding_U_Net,load_model
from Models import DualEncoding_U_Net,load_model,DualEncodingDecoding_U_Net,DualEncoding_U_Net_without_asm,AttnUNet

from PredictNucleiMask import whole_img_pred
from tqdm import tqdm

# Script to test model
dataset='kumar'
model_type='no_asm'

################################################      CoNSep        ###############################################
if dataset=='CoNSeP':
    path_h_gray='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CoNSeP/Test/H_gray'
    path_h_e='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CoNSeP/Test/Images'

################################################      CPM-17        ###############################################
elif dataset=='CPM_17':
    path_h_gray='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CPM_17/Test/H_gray'
    path_h_e='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/CPM_17/Test/Images'

################################################      Kumar        ###############################################
elif dataset=='kumar':
    path_h_gray='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/kumar/processed_data/h_test_dir'
    path_h_e='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Data/kumar/processed_data/h_e_test_dir'

if model_type=='DEAU':
    model=DualEncoding_U_Net(img1_ch=1,img2_ch=1,output_ch=2,dropout=None,include_ffm=False) # Define the model
    input_img2_ch=1
elif model_type=='attn_unet':
    model= AttnUNet(img_ch=3,output_ch=2,dropout=None)
    input_img2_ch=None
    
elif model_type=='no_asm':
    model=DualEncoding_U_Net_without_asm(img1_ch=3,img2_ch=1,output_ch=2,dropout=0.4)
    input_img2_ch=1
    

#Attention and DEAU model dir
#model_CPM_17_final_2020_03_09,model_kumar_final_2020_03_12,model_CoNSep_final_2020_03_15
#model_CPM_17_attn_2020_03_15,model_kumar_attn_2020_03_13,model_CoNSep_attn_2020_03_14

# Ablation study 1) Dual H 
# model_kumar_ab_dual_h_2020_03_19,model_CoNSep_ab_dual_h_2020_03_09,model_CPM_17_ab_dual_h_2020_03_06
# Ablation Study 2) No ASM
# model_CPM_17_ab_no_asm_2020_03_05,model_CoNSep_ab_no_asm_2020_03_11,model_kumar_ab_no_asm_2020_03_18


model_dir='model_kumar_ab_no_asm_2020_03_18'#'model_CoNSep_2020_02_10''model_CPM_17_2020_02_12'
filename='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Trained_models/{}/model_optim.pth'.format(model_dir)       # Path to model weights                                        
pred_dir_name='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/nuclei_seg/Results/prediction_{}_{}'.format('ab_no_asm',dataset)
#'Results/prediction_attn_unet_{}'.format(dataset)               # New directory name where the predicitions are to be
test_list=[x for x in os.listdir(path_h_e) if x.split('.')[-1].lower()=='png']
load_model(filename,model,optimizer=None,scheduler=None)    
print("Model dir :{}\nDataset :{}\nPred dir : {}".format(model_dir,dataset,pred_dir_name))
for img_name in tqdm(test_list):
    h_e_path=os.path.join(path_h_e,img_name)
    h_path=os.path.join(path_h_gray,img_name)
    whole_img_pred(h_e_path,h_path,pred_dir_name,model,input_img1_ch=3,input_img2_ch=input_img2_ch\
                   ,predict_boundary=True,patch_size=256,print_prompt=False)
    
print("DONE")