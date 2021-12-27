import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import Loaders
import Utilities as Util


path_data = '/data/rj21/MyoSeg/Data_ACDC/training'

data_list = []
mask_list = []

# data_list.append('/data/rj21/MyoSeg/Data_MaM/A0S9V9/A0S9V9_sa.nii.gz')
# mask_list.append('/data/rj21/MyoSeg/Data_MaM/A0S9V9/A0S9V9_sa_gt.nii.gz')


for dir_name in os.listdir(path_data):
    if os.path.isdir(os.path.join(path_data, dir_name)):
        
        pat_name = os.path.join(path_data, dir_name)
        # print(pat_path)
        file_name = os.listdir(pat_name)
        
        # for _,file in enumerate(file_name):
        #     data_list.append(os.path.join( pat_path, file ))
            
            # if file.find('gt.nii')>=0:
            # if file.find('frame')==-1:
                # data_list.append(os.path.join( pat_path, file ))
                
        data_list.append( os.path.join(pat_name, file_name[4] ) )
        mask_list.append( os.path.join(pat_name, file_name[5] ) )
        
        # sizeData = Loaders.size_nii( os.path.join( pat_path, file_name[1] ))        
        # print(sizeData)
        
s = np.zeros((100,2))

for pat in range(0,100):
    img_name = data_list[pat]
    mask_name = mask_list[pat]
                           
    sizeData = Loaders.size_nii( img_name )
    
    s[pat,:] = sizeData[0:1]

    # print(sizeData)
    
    # for t in range(0,sizeData[3]):
    #     # for ind_slice in range(0,sizeData[2]):
    #     for ind_slice in range(6,7):
    # for ind_slice in range(0,sizeData[2]):
    #     for t in range(0,sizeData[3]):
    #     # for t in range(0,1):
            # img = Loaders.read_nii(img_name, (0,0,ind_slice,t) )
    
            # plt.Figure()
            # plt.imshow(img, cmap=plt.cm.gray)
            # plt.show()
        
    # for ind_slice in range(0,sizeData[2]):
    for ind_slice in range(3,4):
        img = Loaders.read_nii(img_name, (0,0,ind_slice,0) )
        mask = Loaders.read_nii(mask_name, (0,0,ind_slice,0) )
        # mask = mask==2
        
        # img = Util.crop_min(img)
        # mask = Util.crop_min(mask
                             
        img = Util.center_crop(img, new_width=128, new_height=128)
        mask = Util.center_crop(mask, new_width=128, new_height=128)
        
        plt.Figure()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.imshow(mask, cmap='jet', alpha=0.2)
        plt.show()
        
 