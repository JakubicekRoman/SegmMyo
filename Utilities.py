import numpy as np
import torch
import os
import random
import xlsxwriter
import pandas as pd
import pydicom as dcm
import torchvision.transforms as T
import cv2
import SimpleITK as sitk


# import Loaders


def read_nii(file_name, current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)    
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
    extract_size = (sizeA[0], sizeA[1], 1, 1)
        
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    img = sitk.GetArrayFromImage(file_reader.Execute())
    img = np.squeeze(img)
    
    # img = np.pad(img,((addX[2],addX[3]),(addX[0],addX[1]),(0,0)),'constant',constant_values=(-1024, -1024))


    return img


def size_nii(file_name):
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)    
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
    return sizeA


def augmentation2(img, params):        
    angle = params[0]['Angle']
    translate = params[0]['Transl']
    scale = params[0]['Scale']
    shear = 0
    CenterCrop = params[0]['Crop_size']
    flip = params[0]['Flip']
    
    if flip:
        img = torch.flip(img, [len(img.size())-1])
    
    augm_img = T.functional.affine(img, angle, translate, scale, shear,  T.InterpolationMode('bilinear'))
    augm_img = T.CenterCrop(size=CenterCrop)(augm_img)
    resize = T.Resize((128,128), T.InterpolationMode('bilinear'))
    augm_img = resize(augm_img)
    augm_img = (augm_img - torch.min(augm_img))/ (torch.max(augm_img)-torch.min(augm_img))
    augm_img = T.functional.adjust_sharpness(augm_img, 2)

    return augm_img


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

def crop_center(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2]   
        
    return center_cropped_img

def resize_with_padding(img, expected_size):
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])
    padding[padding<0]=0
    img = np.pad(img, [(padding[0], padding[2]), (padding[1], padding[3])], mode='constant')
    img = crop_center(img, new_width=expected_size[0], new_height=expected_size[1])
    return img


def dice_loss(X, Y):
    eps = 0.00001
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return 1 - dice


def dice_coef(X, Y):
    eps = 0.000001
    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )
    return dice

def MASD_compute(A, B):
    
    A  = A.astype(np.dtype('uint8'))
    A_ctr = A - cv2.dilate(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    distA = cv2.distanceTransform(255-A_ctr, cv2.DIST_L2, 3)
    
    B  = B.astype(np.dtype('uint8'))
    B_ctr = B - cv2.dilate(B, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    
    HD = np.mean(distA[B_ctr>0])  

    return HD


def CreateDataset(path_data, ):      

    data_list_tr = []
    data_list_te = []
    pat = 0

    for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            
            pat_name = os.path.join(path_data, dir_name)
            file_name = os.listdir(pat_name)
            
            sizeData = size_nii( os.path.join(pat_name, file_name[2] ) )
            
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
            
            sizeData = size_nii( os.path.join(pat_name, file_name[1] ) )
            
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
                    sizeData = size_nii( os.path.join( path1, slice_name ) )
                    
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



def CreateDataset_StT_dcm(path_data):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                # if file.find('_W4')>=0:
                    path_mask = os.path.join(path_data, pat_name, file)
                    name = file[0:file.find('_gt')] + file[file.find('_gt')+3:]
                    # path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                    path_maps = os.path.join(path_data, pat_name, name)

                    # sizeData = Loaders.size( path_maps )
                    sizeData = size_nii( path_maps )

                    if len(sizeData)==2:
                        sizeData = sizeData + (1,)
                    # print(sizeData)
                    
                    for sl in range(0,sizeData[2]):
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'slice': name[-7:-4],
                                              'ID_pat': ii,
                                              'ID_scan': iii
                                              } )
                    iii+=1
            
    return data_list_tr

def CreateDataset_StT_UnL_dcm(path_data):
    data_list_tr = []
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        pthX = os.path.join(path_data, pat_name)
        if os.path.isdir(pthX):
            f = os.listdir(pthX)
            for _,sf in enumerate(f):
                pth = os.path.join(path_data, pat_name, sf)
                if os.path.isdir(pth):
                        ff = os.listdir(pth)  
                        for sl,file in enumerate(ff):
                            if file.find('.dcm')>=0:
                                pth2 = os.path.join(path_data, pat_name, sf, file)
                                data_list_tr.append( {'img_path': pth2,
                                                      'mask_path': pth2,
                                                      'pat_name': pat_name,
                                                      'file_name': sf,
                                                      'slice': sl
                                                      # 'ID_pat': ii,
                                                      # 'ID_scan': iii
                                                      } )
                                # iii+=1
            
    return data_list_tr



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
    
 
    
    