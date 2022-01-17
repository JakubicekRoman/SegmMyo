import numpy as np
import torch
import os
import random
import xlsxwriter
import pandas as pd
import pydicom as dcm

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
    eps = 0.00001
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


def CreateDataset_4D(path_data):      

    data_list_tr = []
    pat = 0

    d = os.listdir(path_data)
    for ii in range(88,89):
        dir_name = d[ii]
    # for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            
            pat_name = os.path.join(path_data, dir_name)
            file_name = os.listdir(pat_name)
            
            sizeData = Loaders.size_nii( os.path.join(pat_name, file_name[1] ) )
            
            for ind_slice in range(0,sizeData[2]):
            # for ind_slice in range(0,1):
                for t in range(0,sizeData[3]):
                # for t in range(0,1):
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[1] ), 
                                       'slice': ind_slice, 
                                        'time': t,
                                        'Patient': dir_name } )
            pat = pat+1
            
    return data_list_tr



def CreateDatasetOur(path_data ):      

    data_list = []
    for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            sdir_list = os.listdir(os.path.join(path_data, dir_name))
            sdir_list.sort()
             
            for _, sdir_name in enumerate(sdir_list):
                path1 = os.path.join(path_data, dir_name, sdir_name)
                file_list = os.listdir(path1)
                file_list.sort()
                
                for ind_slice, slice_name in enumerate( file_list ):
                
                    dataset = dcm.dcmread( os.path.join( path1, slice_name ) )
                    t = dataset.SeriesDescription  
                    sizeData = Loaders.size_nii( os.path.join( path1, slice_name ) )
                    
                    if  (np.array(sizeData[0:2])>127).all():
                    #     if t.find('T1')>=0:
                    #         if t.find('post')>=0 or t.find('Post')>=0:
                    #         # if t.find('PRE')>=0 or t.find('Pre')>=0:  
                                
                        if t.find('T2')>=0:
        
                                # ind_slice=0
                                # for slice_name in os.listdir(path1): 
                                    data_list.append( {'img_path': os.path.join(path1, slice_name), 
                                                       'mask_path': 'None',
                                                       'slice': ind_slice,
                                                       'Description': t,
                                                       'Patient': dir_name,
                                                       'Size': sizeData} ) 
                            # ind_slice+=1
            
    # data_list.sort()            
    return data_list



def save_to_excel(dataframe, root_dir, name):
    writer = pd.ExcelWriter(os.path.join(root_dir, '{}.xlsx'.format(name)),
    engine='xlsxwriter',
    datetime_format='yyyy-mm-dd',
    date_format='yyyy-mm-dd')
    sheet = name
    dataframe.to_excel(writer, sheet_name=sheet)
    
    worksheet = writer.sheets[sheet]
    worksheet.set_column('A:ZZ', 22)
    writer.save()
    
    
    