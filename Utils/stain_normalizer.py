import cv2
import glob
import staintools
import os 

norm_brightness = False
imgs_dir = 'Data/CPM_17/Train/Images' 
save_dir = 'Data/CPM_17/Train/Images/norm_python' 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file_list = glob.glob(imgs_dir + '*.png')
file_list.sort() # ensure same order [1]

if norm_brightness:
    standardizer = staintools.BrightnessStandardizer()
stain_normalizer = staintools.StainNormalizer(method='vahadane')

# dict of paths to target image and dir code to make output folder
target_path ='Data/kumar/Tissue_images//TCGA-21-5784-01Z-00-DX1.tif'


target_img = cv2.imread(target_path)
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
if norm_brightness:
    target_img = standardizer.transform(target_img)
stain_normalizer.fit(target_img)
    

for img_path in file_list:
    filename = os.path.basename(img_path)
    basename = filename.split('.')[0]
    print(basename)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm_brightness:
        img = standardizer.transform(img)
    img = stain_normalizer.transform(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("{}/{}.png".format(save_dir, basename), img)