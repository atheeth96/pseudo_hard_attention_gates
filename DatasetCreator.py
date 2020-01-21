import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import skimage
from skimage.io import imread,imsave
from skimage import draw
import imutils
from skimage.filters import threshold_otsu
from skimage.util import pad
from tqdm import tqdm_notebook as tqdm 



def convert_to_png(tif_dir,png_dir):
    images_name=[x for x in os.listdir(tif_dir) if '.tif' in x]
    
    if not os.path.exists(png_dir):
        os.mkdir(png_dir)
        print("made dir {}".format(png_dir))
    for im in tqdm(images_name):
        img_name=im
        img=imread(os.path.join(tif_dir,im))
        img_name=img_name.replace('.tif','.png')    
        imsave(os.path.join(png_dir,img_name),img)
        
        
def rot_image(img):
    test=img.copy()
    rot=test[:, ::-1]
    rot=imutils.rotate(rot,90)
    return rot

def poly2boundry(x,y,img_array):
    if len(x)==2 and len(y)==2:
        rr,cc=draw.line(x[0],y[0],x[1],y[1])
    else:
        rr, cc = draw.polygon_perimeter(x, y)
    img_array[rr,cc]=[255]
    return img_array

def check_in_bounds(x,y,bound):

    if x>=bound:
        x=bound-1
    if y>=bound:
        y=bound-1
    if x<0:
        x=0
    if y<0:
        y=0
    return x,y
        
        
def create_data_vahadane(png_dir,annotation_dir,nucleus_dir,boundary_dir):
    img_name_list=[x for x in os.listdir(png_dir) if '.png' in x]
   
    if not os.path.exists(nucleus_dir):
        os.mkdir(nucleus_dir)
        print("made dir {}".format(nucleus_dir))
    if not os.path.exists(boundary_dir):
        os.mkdir(boundary_dir)
        print("made dir {}".format(boundary_dir))

    

    loop=tqdm(img_name_list)
    for name in loop:
        sample_img=imread(os.path.join(png_dir,name))
        r,c,_=sample_img.shape
        
        xml_name=os.path.join(annotation_dir,name.split('.')[0]+'.xml')
      
        tree=ET.parse(xml_name)
        root=tree.getroot()
        img_nucleus=np.zeros(shape=(r,c),dtype=np.uint8)
        img_boundary=np.zeros(shape=(r,c),dtype=np.uint8)
        
        loop.set_postfix(Regions=len([v.tag for v in root.iter('Vertices')]))

        for v in root.iter('Vertices'):
            X=[]
            Y=[]

            for child in v:
                x=int(eval(child.attrib['X']))
                y=int(eval(child.attrib['Y']))
                x,y=check_in_bounds(x,y,1000)

                X.append(x)
                Y.append(y)


            r_nucleus,c_nucleus=draw.polygon(X,Y)
            img_nucleus[r_nucleus,c_nucleus]=255
            img_nucleus=poly2boundry(X,Y,img_nucleus)
            img_boundary=poly2boundry(X,Y,img_boundary)
            


        img_nucleus.dtype=np.uint8
        img_boundary.dtype=np.uint8
        x,y=np.where(img_boundary==255)
        for i,j in zip(x,y):
            img_boundary[max(i-1,0):min(i+2,r),max(j-1,0):min(j+2,c)]=255
        img_boundary.dtype='uint8'
        
        img_nucleus=rot_image(img_nucleus)
        img_boundary=rot_image(img_boundary)
#         img_boundary=img_boundary-img_nucleus
        
        
        imsave(os.path.join(nucleus_dir,name.split('.')[0]+'_nucleus_mask.png'),img_nucleus)
        imsave(os.path.join(boundary_dir,name.split('.')[0]+'_boundary_mask.png'),img_boundary)
        

    print("DONE")

