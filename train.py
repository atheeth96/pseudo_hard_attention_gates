import numpy as np
import os
import imutils
import pandas as pd
import datetime
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn import metrics
from sklearn.utils import shuffle
from skimage import draw
from skimage.io import imread,imsave
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from skimage.util import pad
import skimage
import time
import math

from tqdm import tqdm 

from Dataset.Dataset import DataSet,Scale,ToTensor,RandomGaussionBlur,RandomMedianBlur\
,RandomHorizontalFlip,RandomRotation,visualize_loader

import warnings
warnings.filterwarnings('ignore')

from Models import DualEncoding_U_Net, save_model,load_model,init_weights,DualEncodingDecoding_U_Net,AttnUNet,\
DualEncoding_U_Net_without_asm,U_Net
from Metrics import SoftDiceLoss,dice_metric,MultiClassBCE,SoftDiceLoss,HV_Loss
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
    
import pickle

## TAGS ##
attention_map=True
pretrained=False
optimizer_selected='adam'
scheduler_type='reduce_on_plateau'
model_type='DualEncoding_U_Net_without_asm'
pretrained_filename='model_CPM_17_v2_2020_02_07/model_optim.pth'
criterion_type='bce'
dataset='CPM_17'
restart_epochs=8
num_epochs=80
predict_boundary=True
assert model_type in ['U-Net','DualEncoding_U_Net','DualEncoding_U_Net_without_asm','DualEncodingDecoding_U_Net','AttnUNet'],\
'Please enter a valid model name'

if model_type in ['U-Net']:
    attention_map=False
    
assert dataset in ['CPM_17','CoNSeP','kumar'],\
'Please enter a valid dataset name'

## DATA DIRECTORY ##
if dataset in ['CPM_17','CoNSep']:
    h_e_train_patch_dir='Data/{}/{}/H_E_patches'.format(dataset,'Train')
    h_train_patch_dir='Data/{}/{}/H_patches'.format(dataset,'Train')

    nuclei_mask_train_patch_dir='Data/{}/{}/nuclei_patches'.format(dataset,'Train')
    boundary_mask_train_patch_dir='Data/{}/{}/boundary_patches'.format(dataset,'Train')
    

    h_e_test_patch_dir='Data/{}/{}/H_E_patches'.format(dataset,'Test')
    h_test_patch_dir='Data/{}/{}/H_patches'.format(dataset,'Test')

    nuclei_mask_test_patch_dir='Data/{}/{}/nuclei_patches'.format(dataset,'Test')
    boundary_mask_test_patch_dir='Data/{}/{}/boundary_patches'.format(dataset,'Test')
    
else:
    

    h_e_train_patch_dir='Data/kumar/processed_data/h_e_train_patch_dir'
    h_train_patch_dir='Data/kumar/processed_data/h_train_patch_dir'
    nuclei_mask_train_patch_dir='Data/kumar/processed_data/nuclei_mask_train_patch_dir'
    boundary_mask_train_patch_dir='Data/kumar/processed_data/boundary_mask_train_patch_dir'

    h_e_test_patch_dir='Data/kumar/processed_data/h_e_test_patch_dir'
    h_test_patch_dir='Data/kumar/processed_data/h_test_patch_dir'
    nuclei_mask_test_patch_dir='Data/kumar/processed_data/nuclei_mask_test_patch_dir'
    boundary_mask_test_patch_dir='Data/kumar/processed_data/boundary_mask_test_patch_dir'


## DEFINE MODEL,LOSS DICTIONARY ###


model_start_date=datetime.datetime.now().strftime("%Y_%m_%d")



model_dict={'U-Net':{'model':U_Net(img_ch=3,output_ch=2,dropout=0.5),\
                     'model_name':os.path.join(os.getcwd(),'Trained_models/model_{}_unet_{}'.format(dataset,model_start_date))},\
            'DualEncoding_U_Net':{'model':DualEncoding_U_Net(img1_ch=3,img2_ch=1,output_ch=2,dropout=0.25,include_ffm=False),\
                     'model_name':os.path.join(os.getcwd(),'Trained_models/model_{}_17_final_{}'.format(dataset,model_start_date))},\
            'DualEncoding_U_Net_without_asm':{'model':DualEncoding_U_Net_without_asm(img1_ch=3,img2_ch=1,output_ch=2,dropout=0.4),\
                     'model_name':os.path.join(os.getcwd(),'Trained_models/model_{}_ab_no_asm_{}'.format(dataset,model_start_date))},\
            'DualEncodingDecoding_U_Net':{'model':DualEncodingDecoding_U_Net(img_ch1=3,img_ch2=1,output_ch1=1,output_ch2=1,dropout=0.45),\
                     'model_name':os.path.join(os.getcwd(),'Trained_models/model_{}_d2_{}'.format(dataset,model_start_date))},\
            'AttnUNet':{'model':AttnUNet(img_ch=3,output_ch=2,dropout=0.5),\
                     'model_name':os.path.join(os.getcwd(),'Trained_models/model_{}_aunet_{}'.format(dataset,model_start_date))},\
            
}

criterion_dict={'bce':nn.BCELoss(),\
                'softdice':SoftDiceLoss(),\
                'multiclassbce':MultiClassBCE(weights=[0.11,0.89])}


## CREATE DATASET ##

batch_size_train=4
batch_size_test=4
                                                
train_transform=torchvision.transforms.Compose([RandomGaussionBlur(p=0.4,sigma=0.5,truncate=4,apply_dual=False),\
                                                RandomMedianBlur(p=0.4,disk_rad=1),\
                                                RandomHorizontalFlip(p=0.4),\
                                                RandomRotation( degrees=[60,120],p=0.38),\
                                                Scale(),\
                                                ToTensor()])

test_transform=torchvision.transforms.Compose([Scale(),ToTensor()])

train_dataset=DataSet(h_e_train_patch_dir,h_train_patch_dir\
                          ,nuclei_mask_train_patch_dir, boundary_mask_train_patch_dir\
                          ,transform=train_transform,attn_gray=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)
test_dataset=DataSet(h_e_test_patch_dir,h_test_patch_dir\
                      ,nuclei_mask_test_patch_dir, boundary_mask_test_patch_dir\
                      ,transform=test_transform,attn_gray=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size_test,shuffle=False)

print(train_loader.__len__())
print(test_loader.__len__())
# img1,img2,img3=visualize_loader(test_loader,0)


## DEFINE MODEL ##

model=model_dict['DualEncoding_U_Net']['model']
BEST_MODEL_PATH=model_dict['DualEncoding_U_Net']['model_name']

if not os.path.exists(BEST_MODEL_PATH):
    os.mkdir(BEST_MODEL_PATH)
    print('{} dir has been made'.format(BEST_MODEL_PATH))
print("Model's state_dict:")
writer = SummaryWriter('{}/experiment_{}'.format(BEST_MODEL_PATH,1))
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    

##DEFINE OPTIM, DEVICE, batchSize, SCHEDULER ##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    

batchsize=batch_size_train
no_steps=train_dataset.__len__()//batchsize

if optimizer_selected=='adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-03, betas=(0.9, 0.98))#,weight_decay=0.02)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-03, momentum=0.8,nesterov=True)


if scheduler_type=='reduce_on_plateau':
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6,\
                                               verbose=True, threshold=0.0001, threshold_mode='rel',\
                                               cooldown=2, min_lr=10e-06, eps=1e-08)
    
else:
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs*no_steps,\
                                                     eta_min=1e-012, last_epoch=-1)

criterion=criterion_dict[criterion_type]

history={'train_loss':[],'test_loss':[],'train_dice':[],'test_dice':[]}
model = model.to(device)


## INITAILIZE MODEL PARAMS ##

if pretrained:
    filename=pretrained_filename
    load_model(filename,model,optimizer=None,scheduler=None)
else:
    init_weights(model)
    
    
## BEGIN TRAINING ##

best_val=0

for epoch in range(num_epochs):
    
    print("Learning Rate : {}".format(optimizer.state_dict()['param_groups'][-1]['lr']))
    # loop over the dataset multiple times
    
    run_avg_train_loss=0
    run_avg_train_dice=0
    
   
    
    run_avg_test_loss=0
    run_avg_test_dice=0
    
    for mode in ['train','eval']:
     
        if mode == 'train':
            
            model.train()
            loop=tqdm(train_loader)
            
            for i, sample_batched in (enumerate(loop)):
                loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
                
                #Clear Gradients
                model.zero_grad()
                
                # get the inputs; data is a list of [dapi, nuclei, boundary]
                if predict_boundary:
                    h_e_train, h_train,nuclei_mask_train,boundary_mask_train = sample_batched['h_e']\
                    ,sample_batched['h']\
                    ,sample_batched['nuclei_mask']\
                    ,sample_batched['boundary_mask']
                    
                    h_e_train, h_train,nuclei_mask_train,boundary_mask_train = \
                    h_e_train.to(device, dtype = torch.float),h_train.to(device, dtype = torch.float)\
                    ,nuclei_mask_train.to(device, dtype = torch.float)\
                    ,boundary_mask_train.to(device, dtype = torch.float)
                    
                    gt_mask_train=torch.cat((nuclei_mask_train,boundary_mask_train),dim=1)
                
                else:
                    h_e_train, h_train,nuclei_mask_train= sample_batched['h_e']\
                    ,sample_batched['h']\
                    ,sample_batched['nuclei_mask']\

                    h_e_train, h_train,nuclei_mask_train= \
                    h_e_train.to(device, dtype = torch.float),h_train.to(device, dtype = torch.float)\
                    ,nuclei_mask_train.to(device, dtype = torch.float)
                    
                    gt_mask_train=nuclei_mask_train

                # forward + backward + optimize
#                 img_batch_comb=torch.cat((img_batch,dapi_batch),dim=1)
                outputs,attn_maps_train = model(h_e_train,h_train)
                outputs=torch.sigmoid(outputs)
    
                if predict_boundary:
                    pred_nuclei_train,pred_boundary_train=torch.chunk(outputs,2,dim=1)
                else:
                    pred_nuclei_train=outputs
                

                
                loss = criterion(outputs, gt_mask_train)
                dice_score=dice_metric(pred_nuclei_train,nuclei_mask_train)
                run_avg_train_loss=(run_avg_train_loss*(0.9))+loss.detach().item()*0.1
                run_avg_train_dice=(run_avg_train_dice*(0.9))+dice_score.detach().item()*0.1
               
                if (i+1)%50==0:
                    
                    if predict_boundary:
                        img_tensor=torch.cat((pred_nuclei_train.detach().cpu(),nuclei_mask_train.detach().cpu(),\
                                             pred_boundary_train.detach().cpu(),boundary_mask_train.detach().cpu()),dim=0)
                    else:
                        img_tensor=torch.cat((pred_nuclei_train.detach().cpu(),nuclei_mask_train.detach().cpu()),dim=0)
                    
                    img_grid2 = torchvision.utils.make_grid(img_tensor,nrow=batch_size_train,padding=10)
                    torchvision.utils.save_image\
                    (img_grid2,os.path.join(BEST_MODEL_PATH,\
                                            'train_iter_{}.png'.format(epoch*len(train_loader)+i+1)))
                    
                    if attention_map:
                    
                        for x,attn_map in enumerate(attn_maps_train):
                            attn_maps_train[x]=attn_maps_train[x].detach().cpu()
                            current_shape=list(attn_maps_train[x].shape)
                            final_shape=list(attn_maps_train[0].shape)


                            if x!=0:
                                pad_shape=((final_shape[-2]-current_shape[-2])//2,\
                                           final_shape[-2]-current_shape[-2]-(final_shape[-2]-current_shape[-2])//2,\
                                          (final_shape[-1]-current_shape[-1])//2,\
                                           final_shape[-2]-current_shape[-2]-(final_shape[-1]-current_shape[-1])//2)
                                attn_maps_train[x]=F.pad(attn_maps_train[x],pad_shape,"constant", 0)



                        img_tensor_attn=torch.cat(tuple(attn_maps_train),dim=0)
                        img_grid_attn = torchvision.utils.make_grid(img_tensor_attn\
                                                                    ,nrow=batch_size_test,padding=10)
                        torchvision.utils.save_image\
                        (img_grid_attn,os.path.join(BEST_MODEL_PATH,\
                                                'attn_train_iter_{}.png'.format(epoch*len(train_loader)+i+1)))
                    
                    writer.add_scalar('Training dice score nuclei',
                            run_avg_train_dice,
                            epoch * len(train_loader) + i+1)
                
                    writer.add_scalar('Training Loss',
                            run_avg_train_loss,
                            epoch * len(train_loader) + i+1)
                    
                loss.backward()
                optimizer.step()
                
                if scheduler_type!='reduce_on_plateau':
                    scheduler.step()
                
                
                loop.set_postfix(loss=run_avg_train_loss,dice_score=run_avg_train_dice)
                
               
            history['train_loss'].append(run_avg_train_loss)
            history['train_dice'].append(run_avg_train_dice)
            
            writer.add_scalar('Train dice { epoch }',
                            run_avg_train_dice,
                            epoch+1)
                
            writer.add_scalar('Train loss { epoch }',
                    run_avg_train_loss,
                    epoch * len(train_loader) + epoch+1)
                
                 
                    
        elif mode =='eval':
            
            #Clear Gradients
            model.zero_grad()
            samples_test=len(test_loader)
            model.eval()
            val_loss=0
            test_agg=0
            for j, test_sample in enumerate(test_loader):
                
                
                if predict_boundary:
                    h_e_test, h_test,nuclei_mask_test,boundary_mask_test = test_sample['h_e']\
                    ,test_sample['h']\
                    ,test_sample['nuclei_mask']\
                    ,test_sample['boundary_mask']
                    
                    h_e_test, h_test,nuclei_mask_test,boundary_mask_test = \
                    h_e_test.to(device, dtype = torch.float),h_test.to(device, dtype = torch.float)\
                    ,nuclei_mask_test.to(device, dtype = torch.float)\
                    ,boundary_mask_test.to(device, dtype = torch.float)
                    
                    gt_mask_test=torch.cat((nuclei_mask_test,boundary_mask_test),dim=1)
                    
                else:
                
                    h_e_test, h_test,nuclei_mask_test= test_sample['h_e']\
                    ,test_sample['h']\
                    ,test_sample['nuclei_mask']

                    h_e_test, h_test,nuclei_mask_test = \
                    h_e_test.to(device, dtype = torch.float),h_test.to(device, dtype = torch.float)\
                    ,nuclei_mask_test.to(device, dtype = torch.float)

                    gt_mask_test=nuclei_mask_test
    
                test_outputs,attn_maps_test = model(h_e_test,h_test)
                test_outputs=torch.sigmoid(test_outputs)
        
        
                if predict_boundary:
                    pred_nuclei_test,pred_boundary_test=torch.chunk(test_outputs,2,dim=1)
                else:
                    pred_nuclei_test=test_outputs
    
                test_loss = criterion(test_outputs, gt_mask_test)
                test_dice=dice_metric(pred_nuclei_test,nuclei_mask_test)
                
                run_avg_test_loss=(run_avg_test_loss*(0.9))+test_loss.detach().item()*0.1
                run_avg_test_dice=(run_avg_test_dice*(0.9))+test_dice.detach().item()*0.1
                
               
                if (j+1)%50==0:
                    if predict_boundary:
                        img_tensor_test=torch.cat((pred_nuclei_test.detach().cpu(),nuclei_mask_test.detach().cpu(),\
                                                  pred_boundary_test.detach().cpu(),boundary_mask_test.detach().cpu()),dim=0)
                    else:
                        img_tensor_test=torch.cat((pred_nuclei_test.detach().cpu(),nuclei_mask_test.detach().cpu()),dim=0)
                    
                    
                    img_grid = torchvision.utils.make_grid(img_tensor_test,nrow=batch_size_test,padding=10)
                    torchvision.utils.save_image\
                    (img_grid,os.path.join(BEST_MODEL_PATH,\
                                            'test_iter_{}.png'.format(epoch*len(test_loader)+j+1)))
                    
                    if attention_map:
                        for x,attn_map in enumerate(attn_maps_test):
                            attn_maps_test[x]=attn_maps_test[x].detach().cpu()
    #                         print(list(attn_maps[0].shape))
                            current_shape=list(attn_maps_test[x].shape)
                            final_shape=list(attn_maps_test[0].shape)
                            if x!=0:
                                pad_shape=((final_shape[-2]-current_shape[-2])//2,\
                                           final_shape[-2]-current_shape[-2]-(final_shape[-2]-current_shape[-2])//2,\
                                          (final_shape[-1]-current_shape[-1])//2,\
                                           final_shape[-2]-current_shape[-2]-(final_shape[-1]-current_shape[-1])//2)
                                attn_maps_test[x]=F.pad(attn_maps_test[x],pad_shape,"constant", 0)


                        img_tensor_attn_test=torch.cat(tuple(attn_maps_test),dim=0)
                        img_grid_attn_test = torchvision.utils.make_grid(img_tensor_attn_test\
                                                                    ,nrow=batch_size_test,padding=10)
                        torchvision.utils.save_image\
                        (img_grid_attn_test,os.path.join(BEST_MODEL_PATH,\
                                                'attn_test_iter_{}.png'.format(epoch*len(train_loader)+j+1)))
                    writer.add_scalar('Testing dice score ',\
                                      run_avg_test_dice,epoch * len(test_loader) + j+1)
                    
                    writer.add_scalar('Testing Loss',\
                                      run_avg_test_loss,epoch * len(test_loader) + j+1)
                
            print("test_loss: {}\ntest_dice :{}"\
                  .format(run_avg_test_loss,run_avg_test_dice))
            history['test_loss'].append(run_avg_test_loss)
            history['test_dice'].append(run_avg_test_dice)
            
            writer.add_scalar('Test dice { epoch }',
                            run_avg_test_dice,
                            epoch+1)
                
            writer.add_scalar('Test loss { epoch }',
                    run_avg_test_loss,
                    epoch * len(train_loader) + epoch+1)
            
            if run_avg_test_dice>best_val:
                best_val=run_avg_test_dice
                save_model(model,optimizer,BEST_MODEL_PATH+\
                           '/model_optim.pth',scheduler=scheduler)
                print("saved model with test dice score: {}".format(best_val))
            if scheduler_type=='reduce_on_plateau':
                scheduler.step(run_avg_test_loss)
    
save_model(model,optimizer, BEST_MODEL_PATH+'/model_final.pth',scheduler=scheduler)    

with open("{}/history.txt".format(BEST_MODEL_PATH), "wb") as fp:   #Pickling
    pickle.dump(history, fp)