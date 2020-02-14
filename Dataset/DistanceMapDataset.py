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

    def __init__(self, h_e_dir,h_dir,nuclei_mask_dir,hor_dir,ver_dir, transform=None,attn_gray=True):
        
       
        self.h_e_dir=h_e_dir
        self.h_dir = h_dir
        
        self.nuclei_mask_dir=nuclei_mask_dir
        self.hor_dir=hor_dir
        self.ver_dir=ver_dir
        
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
        hor_path = os.path.join(self.hor_dir,
                                self.img_list[idx])
        
        ver_path = os.path.join(self.ver_dir,
                                self.img_list[idx])
    
 
       
        h_e = imread(h_e_img_path)
        
        
        if self.attn_gray:
            h = np.expand_dims(imread(h_img_path),axis=2)
            
        else:
            h =imread(h_img_path)
            
        h=np.amax(h)-h
         
        
        
        nuclei_mask=np.expand_dims(imread(nuclei_mask_path),axis=2)
        hor_map=np.expand_dims(imread(hor_path),axis=2)
        ver_map=np.expand_dims(imread(ver_path),axis=2)
        
        sample={'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'hor_map':hor_map,\
               'ver_map':ver_map}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class Scale(object):
    

    def __call__(self,sample):
       
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']
        
 
        scale=255
        h=h/scale
        h_e=h_e/scale
        nuclei_mask=nuclei_mask/scale
        hor_map=hor_map/scale
        ver_map=ver_map/scale
#         print("Scale : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))

        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'hor_map':hor_map,\
               'ver_map':ver_map}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        h_e = h_e.transpose((2, 0, 1))
        h = h.transpose((2, 0, 1))
        
        nuclei_mask = nuclei_mask.transpose((2, 0, 1))
        hor_map = hor_map.transpose((2, 0, 1))
        ver_map = ver_map.transpose((2, 0, 1))
        
        
#         print("ToTensor : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        
        return {'h_e': torch.from_numpy(h_e).type(torch.FloatTensor),\
                'h':torch.from_numpy(h).type(torch.FloatTensor),\
                'nuclei_mask':torch.from_numpy(nuclei_mask).type(torch.FloatTensor),\
                'hor_map':torch.from_numpy(hor_map).type(torch.FloatTensor),\
               'ver_map':torch.from_numpy(ver_map).type(torch.FloatTensor)}
    
    
class RandomGaussionBlur(object):
    """Apply gaussian blur to ndarrays in sample."""
    def __init__(self,p,sigma=1,truncate=3,apply_dual=True):
        self.p=p
        self.sigma=sigma
        self.truncate=truncate
        self.apply_dual=apply_dual
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']

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
                'hor_map':hor_map,\
               'ver_map':ver_map}
    
    
class RandomMedianBlur(object):
    """Apply gaussian blur to ndarrays in sample."""
    def __init__(self,p=0.2,disk_rad=2,apply_dual=False):
        self.p=p
        self.disk_rad=disk_rad
        self.apply_dual=apply_dual
    

    def __call__(self, sample):
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']

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
        
#         print("RandomMedianBlur : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'hor_map':hor_map,\
               'ver_map':ver_map}
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']
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
            hor_map= hor_map[ ::-1,:]
            ver_map= ver_map[ ::-1,:]
            
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        if np.amax(nuclei_mask)<=1:
            nuclei_mask=(nuclei_mask*255).astype(np.uint8)
        if np.amax(hor_map)<=1:
            hor_map=(hor_map*255).astype(np.uint8)
        if np.amax(ver_map)<=1:
            ver_map=(ver_map*255).astype(np.uint8)
        
#         print("RandomHorizontalFlip : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'hor_map':hor_map,\
               'ver_map':ver_map}
    
    
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
        
        h_e,h,nuclei_mask,hor_map,ver_map=sample['h_e'],sample['h'],sample['nuclei_mask'],sample['hor_map'],sample['ver_map']
        
        if random.random() < self.p:
        
            angle = self.get_params(self.degrees)
            
            h_e= rotate(h_e, angle)*255
            
            h= rotate(h, angle)*255
            
            nuclei_mask= rotate(nuclei_mask, angle)
            ver_map= rotate(ver_map, angle)
            hor_map= rotate(hor_map, angle)
        
        if np.amax(h_e)<=1:
            h_e=(h_e*255).astype(np.uint8)
        if np.amax(h)<=1:
            h=(h*255).astype(np.uint8)
        if np.amax(nuclei_mask)<=1:
            nuclei_mask=(nuclei_mask*255).astype(np.uint8)
        if np.amax(hor_map)<=1:
            hor_map=(hor_map*255).astype(np.uint8)
        if np.amax(ver_map)<=1:
            ver_map=(ver_map*255).astype(np.uint8)
            
#         print("RandomRotation : ",np.amax(h_e),np.amax(h),np.amax(nuclei_mask),np.amax(boundary_mask))
        return {'h_e': h_e,\
                'h':h,\
                'nuclei_mask':nuclei_mask,\
                'hor_map':hor_map,\
               'ver_map':ver_map}

        

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
            ver_map=(sample['ver_map'][index]).numpy()
            hor_map=(sample['hor_map'][index]).numpy()
            
            print("MAX VALUE : ","\nH&E",np.amax(h_e),"\nH",np.amax(h),"\nnuclei_mask",\
                  np.amax(nuclei_mask),"\nhor_map",np.amax(hor_map),"\nver_map",np.amax(ver_map))
            
            
            h_e=h_e.transpose(1,2,0)
            
            h=np.squeeze(h.transpose(1,2,0),axis=2)
            
            nuclei_mask=np.squeeze(nuclei_mask.transpose(1,2,0),axis=2)
            hor_map=np.squeeze(hor_map.transpose(1,2,0),axis=2)
            ver_map=np.squeeze(ver_map.transpose(1,2,0),axis=2)
            
            fig=plt.figure()
            plt.imshow(h_e)
            
            fig=plt.figure()
            plt.imshow(h,cmap='gray')
            
            fig=plt.figure()
            plt.imshow(nuclei_mask,cmap='gray')
            
            fig=plt.figure()
            plt.imshow(hor_map,cmap='gray')
            
            fig=plt.figure()
            plt.imshow(ver_map,cmap='gray')

            break
