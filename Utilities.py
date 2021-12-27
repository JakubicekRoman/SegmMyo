import numpy as np
import torch
import os
import random

import Loaders


def augmentation(img, new_width=None, new_height=None, rand_tr='Rand'):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)  

    if isinstance(rand_tr, str):
        if rand_tr.find('Rand')>=0:
            max_tr = (int(np.ceil((width - new_width) / 4)), int(np.ceil((height - new_height) / 4))) 
            rand_transl = (random.randint( -max_tr[0], max_tr[0]) , random.randint( -max_tr[1], max_tr[1] )  )
            # print(rand_transl)
        elif rand_tr.find('None')>=0:
            rand_transl = (0, 0)
    else:
        rand_transl = rand_tr
        
    
    left = int(np.ceil((width - new_width) / 2)) + rand_transl[0]
    right = width - int(np.floor((width - new_width) / 2)) + rand_transl[0]

    top = int(np.ceil((height - new_height) / 2)) + rand_transl[1]
    bottom = height - int(np.floor((height - new_height) / 2)) + rand_transl[1]
    
    # left = int(np.ceil((width - new_width) / 2))
    # right = width - int(np.floor((width - new_width) / 2))

    # top = int(np.ceil((height - new_height) / 2))
    # bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img, rand_transl



def crop_min(img):
    
    s = min( img.shape)  
    img = img[0:s,0:s]
    
    return img


def dice_loss(X, Y):
    eps = 0.000
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice

def dice_coef(X, Y):
    eps = 0.000
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return dice


def CreateDataset(path_data, ):      

    data_list_tr = []
    data_list_te = []
    pat = 0

    for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            
            pat_name = os.path.join(path_data, dir_name)
            file_name = os.listdir(pat_name)
            
            sizeData = Loaders.size_nii( os.path.join(pat_name, file_name[2] ) )
            
            if pat<=80:
                for ind_slice in range(0,sizeData[2]):
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[2] ), 
                                       'mask_path': os.path.join(pat_name, file_name[3] ),
                                       'slice': ind_slice } )
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                       'mask_path': os.path.join(pat_name, file_name[5] ),
                                       'slice': ind_slice } )
            else:
                for ind_slice in range(0,sizeData[2]):
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[2] ), 
                                       'mask_path': os.path.join(pat_name, file_name[3] ),
                                       'slice': ind_slice } )    
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                       'mask_path': os.path.join(pat_name, file_name[5] ),
                                       'slice': ind_slice } )
            pat = pat+1
            
    return data_list_tr, data_list_te