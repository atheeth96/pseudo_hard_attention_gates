from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt



class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, h_e_dir,h_dir,nuclei_mask_dir, boundary_mask_dir, transform=None,attn_gray=True):
        
       
        self.h_e_dir=h_e_dir
        self.h_dir = h_dir
        
        self.nuclei_mask_dir=nuclei_mask_dir
        self.boundary_mask_dir=boundary_mask_dir
        
        self.transform = transform
        self.attn_gray=attn_gray
  
        self.img_list=[x for x in os.listdir(self.h_e_dir) if x.split('.')[-1]=='png']
#Returns length of data-set unlike its keras counter part that returns no_batches
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        h_e_img_path = os.path.join(self.h_e_dir,
                                self.img_list[idx])
        h_img_path = os.path.join(self.h_dir,
                                self.img_list[idx])
        
        nuclei_mask_path = os.path.join(self.nuclei_mask_dir,
                                self.img_list[idx])
        boundary_mask_path = os.path.join(self.boundary_mask_dir,
                                self.img_list[idx])
    
 
       
        h_e_img = imread(h_e_img_path)
        
        
        if self.attn_gray:
            h_img = np.expand_dims(imread(h_img_path),axis=2)
            
        else:
            h_img =imread(h_img_path)
        
        
        nuclei_mask=np.expand_dims(imread(nuclei_mask_path),axis=2)
        boundary_mask=np.expand_dims(imread(boundary_mask_path),axis=2)
        
        sample={'h_e': h_e_img,\
                'h':h_img,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class Scale(object):
    

    def __call__(self,sample):
       
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']
        
 
        scale=255

        return {'h_e': h_e/scale,\
                'h':h/scale,\
                'nuclei_mask':nuclei_mask/scale,\
                'boundary_mask':boundary_mask/scale}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        h_e = h_e.transpose((2, 0, 1))
        h = h.transpose((2, 0, 1))
        
        nuclei_mask = nuclei_mask.transpose((2, 0, 1))
        boundary_mask = boundary_mask.transpose((2, 0, 1))
        
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}
    


class  Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    
    
    
def visualize_loader(loader,index=0):
    for i,sample in enumerate(loader):
        #print(sample['image'].shape)
        if i==1:
           
            h_e=(sample['h_e'][index]).numpy()
            h=(sample['h'][index]).numpy()
            
            nuclei_mask=(sample['nuclei_mask'][index]).numpy()
            boundary_mask=(sample['boundary_mask'][index]).numpy()
            
            h_e=h_e.transpose(1,2,0)
            h=np.squeeze(h.transpose(1,2,0),axis=2)
            
            nuclei_mask=np.squeeze(nuclei_mask.transpose(1,2,0),axis=2)
            boundary_mask=np.squeeze(boundary_mask.transpose(1,2,0),axis=2)
            
            fig=plt.figure()
            plt.imshow(h_e)
            
            fig=plt.figure()
            plt.imshow(h,cmap='gray')
            
            fig=plt.figure()
            plt.imshow(nuclei_mask,cmap='gray')
            
            fig=plt.figure()
            plt.imshow(boundary_mask,cmap='gray')

            break
