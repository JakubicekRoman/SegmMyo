import os
import numpy as np
# import matplotlib.pyplot as plt
# import SimpleITK as sitk
import torch
# from torch.utils import data
# import torch.optim as optim
import glob
import random
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
# # path_data = '/data/rj21/Data/Test_data/example_data_joint_sep'  # Linux bioeng358
# # path_data = '/data/rj21/Data/Test_data/Clin_Unl'  # Linux bioeng358
# path_data = '/data/rj21/Data/Test_data/T2_Alinas'
# # path_save = '/data/rj21/Data/Test_data/Res_CLin'
# path_save = '/data/rj21/Data/Test_data/Res_T2_Alinas'
# # vNet = '/data/rj21/MyoSeg/Models/net_v9_2_1.pt'
# vNet = '/data/rj21/MyoSeg/Models/net_v9_1_6_6.pt'

# vNet = '/data/rj21/MyoSeg/Models/net_v8_3_8.pt'
# path_data = '/data/rj21/Data/Test_data/T2_Alinas'
# path_save = '/data/rj21/Data/Test_data/T2_Alinas_res'

# vNet = '/data/rj21/MyoSeg/Models/net_v8_3_9_1.pt'
# path_data = '/data/rj21/Data/Test_data/Clin_test'
# path_save = '/data/rj21/Data/Test_data/Clin_test_res'

def Predict(path_data, path_save, vNet, bagging=True):
        
    if vNet.find('net_v8')>=0:
        vel_cut = 128
        
    if vNet.find('net_v8_3')>=0 or vNet.find('net_v8_4')>=0:
        vel_cut = 256
        
    if vNet.find('net_v9_')>=0:
        vel_cut = 256
        import Network_v9 as Network
    else:
        import Network as Network
    
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
        
        RescaleSlope=1
        if len(dataset.dir('RescaleSlope'))>0:
            RescaleSlope = float(dataset['RescaleSlope'].value)
        RescaleIntercept=1
        if len(dataset.dir('RescaleIntercept'))>0:
            RescaleIntercept = float(dataset['RescaleIntercept'].value)   
        img = img*RescaleSlope + RescaleIntercept
    
    
        if len(dataset.dir('PixelSpacing'))>0:
            resO = (dataset['PixelSpacing'].value[0:2])
        else:
            # resO = (2.0, 2.0)
            resO = (1.0, 1.0) 
            
        resN = (1.0, 1.0) 
        img = torch.tensor(np.expand_dims(img, [0]).astype(np.float32))
        img = Util.Resampling(img, resO, resN, 'bilinear').detach().numpy()[0,:,:]
        
        imgOrig = img.copy()
        vel = np.shape(img)
            
        if bagging:
            rot = [45,95,120,170,187,230,250,275,0]
        else:
            rot = [0]
        
        Res = np.zeros((np.size(rot),vel[0],vel[1]))
        k=0
        for r in rot:
            
            if rot == 0:
                transl=0; flip=False
            else:
                transl = (random.randint(-20/resN[0],20/resN[0]),random.randint(-20/resN[0],20/resN[0]))
                flip = random.uniform(0, 1)>=0.5
                    
            img = Util.rot_transl(imgOrig, r, transl, flip, invers=False)   
        
            img, p_cut, p_pad = Util.crop_center_final(img, new_width=vel_cut, new_height=vel_cut)
            
            img = torch.tensor(np.expand_dims(img, [0,1]).astype(np.float32))
            
            with torch.no_grad(): 
                res = net( img.cuda() )
                if vNet.find('net_v9_')>=0:
                    res = torch.sigmoid(res)
                else:
                    res = torch.softmax(res,dim=1)
            
            res = res[0,0,:,:].detach().cpu().numpy()>0.5
            
            # plt.figure
            # plt.imshow(res,cmap='jet')
            # plt.show()
            
            velR = np.shape(res)
            res = res[p_pad[0]:velR[0]-p_pad[1],p_pad[2]:velR[1]-p_pad[3]]
            
            res1 = np.zeros(vel,dtype='uint16')
            res1[p_cut[0]:p_cut[1],p_cut[2]:p_cut[3]] = res
        
            res1 = Util.rot_transl(res1, r, transl, flip, invers=True)   
            
            Res[k,:,:] = res1
            k=k+1
        
        # plt.figure
        # plt.imshow(imgOrig,cmap='gray')
        # plt.imshow(res1,cmap='cool', alpha=0.1)
        # plt.show()
        
        if bagging:
            res = np.median(Res,0) > 0.5
        else:
            res = Res[-1,:,:]
        
        dataset.PixelData = res
     
        full_path_save = str( path_save +  file[len(path_data):-4] + '_mask')    
        full_path_save = full_path_save.replace('.', '')
        full_path_save = full_path_save.replace('-', '')
        
        dir_path = full_path_save[0:full_path_save.rfind('/')]+'/'
        if not os.path.exists(dir_path):
            # shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        
            
        dataset.save_as(full_path_save + '.dcm')
        
        mdic = {"segm_mask": res, "dcm_data": imgOrig, "prob_maps": Res}
        savemat( str(full_path_save +'.mat'), mdic)
    
        # Util.progress(i, len(data_list), status='in progress')
        bar_len = 5
        filled_len = int(round(bar_len * i / float(len(data_list))))
        # print(filled_len)
        # print(filled_len_old)
        if not (int(filled_len) == int(filled_len_old)):
            print( "%.0f" % ( i/len(data_list)*100 ) + '%')
            filled_len_old = filled_len
     
    print( '100% ... done' )
        
        
def PredictFour(path_data, path_save, vNet):
            
    if vNet.find('net_v8')>=0:  
        vel_cut = 128
        
    if vNet.find('net_v8_3')>=0 or vNet.find('net_v8_4')>=0:
        vel_cut = 256
        
    if vNet.find('net_v9_')>=0:
        vel_cut = 256
        import Network_v9 as Network
    else:
        import Network as Network
        
    
    data_list = glob.glob(os.path.normpath( path_data + '/**/T1/*.dcm' ), recursive=True)
    # data_list = data_list[100:101]
    # data_list = data_list[0:500:100]
    
    print('\n initializing ...')
    filled_len_old=-1
    
    Imgs = torch.tensor(np.zeros((1,4,vel_cut,vel_cut) ), dtype=torch.float32)
    
    for i in range(0,len(data_list)):
        file1 = data_list[i]
        # nextSub = file[len(path_data):]
        
        net = torch.load(vNet)
        net = net.cuda()
        
        nImg = ('T1','T2','W1','W4')      
        for c in range(0,4):
            
            file  = file1.replace('/T1/','/' + nImg[c] + '/')
            dataset = dcm.dcmread(file)
            img = dataset.pixel_array
            img_orig = img.copy()
            
            RescaleSlope=1
            if len(dataset.dir('RescaleSlope'))>0:
                RescaleSlope = float(dataset['RescaleSlope'].value)
            RescaleIntercept=1
            if len(dataset.dir('RescaleIntercept'))>0:
                RescaleIntercept = float(dataset['RescaleIntercept'].value)
                   
            img = img*RescaleSlope + RescaleIntercept
            # imgOrig = img.copy()
            
            vel = np.shape(img)
            img, p_cut, p_pad = Util.crop_center_final(img, new_width=vel_cut, new_height=vel_cut)
            
            Imgs[0,c,:,:] = torch.tensor(img.astype(np.float32))
        
        with torch.no_grad(): 
            res = net( Imgs.cuda() )
            if vNet.find('net_v9_')>=0:
                res = torch.sigmoid(res)
            else:
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
        
        save_file =  file[len(path_data):-4].replace('/W4/','/')
        full_path_save = str( path_save + save_file + '_mask')
    
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
    

# Predict(path_data, path_save, vNet, bagging=True)
