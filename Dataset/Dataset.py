from skimage.io import imread
from skimage.filters import gaussian,median
from skimage.transform import rotate
from skimage.morphology import disk
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
    """H&E and H dataset."""

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
            
        h_img=np.amax(h_img)-h_img
         
        
        
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
        h=h/scale
        h_e=h_e/scale
        nuclei_mask=nuclei_mask/scale
        boundary_mask=boundary_mask/scale
#         print("Scale : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))

        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}

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
        
        
#         print("ToTensor : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        
        return {'h_e': torch.from_numpy(h_e).type(torch.FloatTensor),\
                'h':torch.from_numpy(h).type(torch.FloatTensor),\
                'nuclei_mask':torch.from_numpy(nuclei_mask).type(torch.FloatTensor),\
                'boundary_mask':torch.from_numpy(boundary_mask).type(torch.FloatTensor)}
    
    
class RandomGaussionBlur(object):
    """Apply gaussian blur to ndarrays in sample."""
    def __init__(self,p,sigma=1,truncate=3,apply_dual=True):
        self.p=p
        self.sigma=sigma
        self.truncate=truncate
        self.apply_dual=apply_dual
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if random.random() < self.p:
            if self.apply_dual:
                h_e= skimage.filters.gaussian(h_e, sigma=self.sigma, output=None, \
                           mode='nearest', cval=0, multichannel=None, \
                           preserve_range=False, truncate=self.truncate)
                h= skimage.filters.gaussian(h, sigma=self.sigma, output=None, \
                           mode='nearest', cval=0, multichannel=None, \
                           preserve_range=False, truncate=3)
            else:
                h_e= skimage.filters.gaussian(h_e, sigma=self.sigma, output=None, \
                           mode='nearest', cval=0, multichannel=None, \
                           preserve_range=False, truncate=self.truncate)
                
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        
            
#         print("RandomGaussionBlur : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))

        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}
    
    
class RandomMedianBlur(object):
    """Apply gaussian blur to ndarrays in sample."""
    def __init__(self,p=0.2,disk_rad=2,apply_dual=False):
        self.p=p
        self.disk_rad=disk_rad
        self.apply_dual=apply_dual
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if random.random() < self.p:
            if self.apply_dual:
                h_e= median(h_e, selem=np.stack((disk(self.disk_rad),disk(self.disk_rad),disk(self.disk_rad)),axis=2))
                h= median(h, disk(self.disk_rad))
            else:
                h_e= median(h_e, selem=np.stack((disk(self.disk_rad),disk(self.disk_rad),disk(self.disk_rad)),axis=2))
            
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        if np.amax(nuclei_mask)<=1:
            nuclei_mask=(nuclei_mask*255).astype(np.uint8)
        if np.amax(boundary_mask)<=1:
            boundary_mask=(boundary_mask*255).astype(np.uint8)
            
#         print("RandomMedianBlur : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}
    
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']
        """
        Args:
            sample 

        Returns:
             Image: Randomly flipped image.
        """
        if random.random() < self.p:
            h_e= h_e[ ::-1,:]
            h= h[ ::-1,:]
            nuclei_mask= nuclei_mask[ ::-1,:]
            boundary_mask= boundary_mask[ ::-1,:]
            
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        if np.amax(nuclei_mask)<=1:
            nuclei_mask=(nuclei_mask*255).astype(np.uint8)
        if np.amax(boundary_mask)<=1:
            boundary_mask=(boundary_mask*255).astype(np.uint8)
        
#         print("RandomHorizontalFlip : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}
    
    
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).

    """

    def __init__(self, degrees=[60,120],p=1):
        
        self.degrees = degrees
        self.p=p

        

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        
        h_e,h,nuclei_mask,boundary_mask=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['boundary_mask']
        
        if random.random() < self.p:
        
            angle = self.get_params(self.degrees)
            
            h_e= rotate(h_e, angle)*255
            
            h= rotate(h, angle)*255
            
            nuclei_mask= rotate(nuclei_mask, angle)
            boundary_mask= rotate(boundary_mask, angle)
        
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        if np.amax(nuclei_mask)<=1:
            nuclei_mask=(nuclei_mask*255).astype(np.uint8)
        if np.amax(boundary_mask)<=1:
            boundary_mask=(boundary_mask*255).astype(np.uint8)
            
#         print("RandomRotation : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'boundary_mask':boundary_mask}

        

        return 

    


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
    
    
# Function to visualize data loader 

def visualize_loader(loader,index=0):
    for i,sample in enumerate(loader):
        #print(sample['image'].shape)
        if i==1:
           
            h_e=(sample['h_e'][index]).numpy()
            h=(sample['h'][index]).numpy()
            
            
            nuclei_mask=(sample['nuclei_mask'][index]).numpy()
            boundary_mask=(sample['boundary_mask'][index]).numpy()
            print("MAX VALUE : ","\nH&E",np.amax(h_e),"\nH",np.amax(h),"\nnuclei_mask",\
                  np.amax(nuclei_mask),"\nboundary_mask",np.amax(boundary_mask))
            
            
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
