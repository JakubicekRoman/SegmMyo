## dataloader for new data2 - annotated

#for training Unet of ACDC - ED, ES
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import glob
import random
import torchvision.transforms as T
import pandas as pd
import cv2


import Loaders
import Unet_2D


def CreateDataset7(path_data):      
    data_list_tr = []
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                path_mask = os.path.join(path_data, pat_name, file)
                name = file[0:file.find('_gt')]
                path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                
                sizeData = Loaders.size_nii( path_maps )
                if len(sizeData)==2:
                    sizeData = sizeData + (1,)
                print(sizeData)
                
                for sl in range(0,sizeData[2]):
                    data_list_tr.append( {'img_path': path_maps,
                                          'mask_path': path_mask,
                                          'pat_name': pat_name,
                                          'file_name': name,
                                          'Slice': sl } )
            
    return data_list_tr


## -------------- validation for \StT data annotated ------------------
path_data = '/data/rj21/Data/Data2/Resaved_data_StT'  # Linux bioeng358
# path_data = 'D:\jakubicek\SegmMyo\Data_ACDC\\training'  # Win CUDA2
data_list_train = CreateDataset7(os.path.normpath( path_data ))




