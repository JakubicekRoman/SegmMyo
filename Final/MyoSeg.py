import os
import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
import torch
# from torch.utils import data
# import torch.optim as optim
import glob
# import random
# import torchvision.transforms as T
# import pandas as pd
# import cv2
# import pickle
import pydicom as dcm
from scipy.io import savemat

import Utilities as Util
# import Loaders
# import Unet_2D


## -------------- validation for \StT data annotated ------------------
# path_data = '/data/rj21/Data/Test_data/example_data'  # Linux bioeng358
# path_save = '/data/rj21/MyoSeg/Final/Results'

def Predict(path_data, path_save, vNet):
    
    
    data_list = glob.glob(os.path.normpath( path_data + '/**/*.dcm' ), recursive=True)
    # data_list = data_list[100:101]
    # data_list = data_list[0:500:100]
    
    print('\n initializing ...')
    filled_len_old=-1
    
    for i in range(0,len(data_list)):
        file = data_list[i]
        # nextSub = file[len(path_data):]
        
        net = torch.load(vNet)
        net = net.cuda()
        
        dataset = dcm.dcmread(file)
        img = dataset.pixel_array
        img_orig = img.copy()
        # plt.figure
        # plt.imshow(img,cmap='jet')
        # plt.show()
        
        RescaleSlope=1
        if len(dataset.dir('RescaleSlope'))>0:
            RescaleSlope = float(dataset['RescaleSlope'].value)
        RescaleIntercept=1
        if len(dataset.dir('RescaleIntercept'))>0:
            RescaleIntercept = float(dataset['RescaleIntercept'].value)
               
        img = img*RescaleSlope + RescaleIntercept
        imgOrig = img.copy()
        
        vel = np.shape(img)
        img, p_cut, p_pad = Util.crop_center_final(img, new_width=128, new_height=128)
        
        # plt.figure
        # plt.imshow(img,cmap='gray')
        # plt.show()
        
        img = torch.tensor(np.expand_dims(img, [0,1]).astype(np.float32))
        
        with torch.no_grad(): 
            res = net( img.cuda() )
            res = torch.softmax(res,dim=1)
        
        res = res[0,0,:,:].detach().cpu().numpy()>0.5
        
        # plt.figure
        # plt.imshow(res,cmap='jet')
        # plt.show()
        
        velR = np.shape(res)
        res = res[p_pad[0]:velR[0]-p_pad[1],p_pad[2]:velR[1]-p_pad[3]]
        
        res1 = np.zeros(vel,dtype='uint16')
        res1[p_cut[0]:p_cut[1],p_cut[2]:p_cut[3]] = res
        
        # plt.figure
        # plt.imshow(imgOrig,cmap='gray')
        # plt.imshow(res1,cmap='cool', alpha=0.1)
        # plt.show()
        
        dataset.PixelData = res1
        
        full_path_save = str( path_save +  file[len(path_data):-4] + '_mask')
    
        full_path_save = full_path_save.replace('.', '')
        full_path_save = full_path_save.replace('-', '')
        
        dir_path = full_path_save[0:full_path_save.rfind('/')]+'/'
        if not os.path.exists(dir_path):
            # shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        
            
        dataset.save_as(full_path_save + '.dcm')
        
        mdic = {"segm_mask": res1, "dcm_data": img_orig}
        savemat( str(full_path_save +'.mat'), mdic)
    
        # Util.progress(i, len(data_list), status='in progress')
        bar_len = 5
        filled_len = int(round(bar_len * i / float(len(data_list))))
        # print(filled_len)
        # print(filled_len_old)
        if not (int(filled_len) == int(filled_len_old)):
            print( "%.2f" % ( i/len(data_list)) + '%')
            filled_len_old = filled_len
     
    print( '1.00% ... done' )