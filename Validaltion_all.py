#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:18:08 2022

@author: rj21
"""
## dataloader for new data2 - annotated

# for validation

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
import pickle
import pydicom as dcm
from skimage import measure, morphology
from scipy.io import savemat
import nibabel as nib
from sklearn.utils import shuffle

import Utilities as Util
import Loaders
# import Network_Att as Network
import Network as Network
#
data_list_test=[]


path_data = '/data/rj21/MyoSeg/params_net' + os.sep + 'scans_list_Task744_MyoSeg.xlsx'

data_list_test = pd.read_excel(path_data)


# seznam = ('v10_4_4x','v10_4_4xx','v10_5_1_end','v10_5_1','v10_5_2_end','v10_5_2')
# seznam = ('v10_6_1', 'v10_6_1_end')
# seznam = ('v10_6_1_end',)
# seznam = ('v12_0_1',)
seznam = ('v12_0_1','v12_0_1_end','v12_0_2','v12_0_2_end','v12_1_1','v12_1_1_end')


# for v in range(0,2):
for _, version in enumerate(seznam):

    # random.seed(777)
    
    
# version = 'v11_0_3'
# version = 'v10_1_3'
# version = 'v10_4_1'
    
    # version = version + '_seed77'
    
    net = torch.load(r"/data/rj21/MyoSeg/Models/net_" + version + ".pt")
    net = net.cuda()
        
    # path_save = '/data/rj21/MyoSeg/valid/expert/cine/'
    
    path_save = '/data/rj21/MyoSeg/valid/valid_002/'
    path_save_data = path_save + os.sep + version    
    # version = version + ""
    # version = version + "_seed77"
    
    # save_name = 'Joint_valid'
    # save_name = 'All_valid'
    # save_name = 'alinas_valid'
    # save_name = '_test'
    save_name = '_all'
    
    try:
        os.mkdir(path_save)
    except OSError as error:
        print(error) 
        
    # try:
    #     os.mkdir(path_save_data)
    # except OSError as error:
    #     print(error)     
    
    res_table = pd.DataFrame(data=[], columns=['Name','Set','Slice' ,'Dataset','Seq'] )                                         
    
    diceTe=[]
    vel=[]
    iii=0
    velImg = 256;
    # velImg = 128;
    
    params = (velImg,  velImg,velImg,  0,0,  0,0,0,0,   1.0,1.0,  1.0) 
    
    # random.seed(77)
    
    # data_list_test = shuffle(data_list_test, random_state=777)
    
    # for num in range(0,20):
    for num in range(0,len(data_list_test),1):
    # ind=(0,50)
    # for num in ind:
    
        Imgs = torch.tensor(np.zeros((1,1,velImg,velImg) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((1,1,velImg,velImg) ), dtype=torch.float32)
    
        current_index = data_list_test.iloc[num]['Slice']
        img_path = data_list_test.iloc[num]['Data_path']
        mask_path = img_path.replace('Data.','Mask.')
        nPat = data_list_test.iloc[num]['Patient']

        t=0
        if img_path.find('.nii')>0:
            nii_data = nib.load(img_path)
            img = nii_data.dataobj[:,:,current_index]
            nii_dataM = nib.load(mask_path)
            mask = nii_dataM.dataobj[:,:,current_index]
            # mask = mask==1
        elif img_path.find('.dcm')>0:
            dataset = dcm.dcmread(img_path)
            img = dataset.pixel_array.astype(dtype='float32')
            dataset = dcm.dcmread(mask_path)
            mask = dataset.pixel_array
            mask = mask==1  
        
        resO = tuple(nii_data.header['pixdim'][1:4])
        resN = (params[11],params[11])
            
        vel.append(img.shape)
    
        img = torch.tensor(np.expand_dims(img, 0).astype(np.float32))
        mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32))
    
        
        img = Util.Resampling(img, resO, resN, 'bilinear')
        mask = Util.Resampling(mask, resO,  resN, 'nearest')
        imgO = img;
        
        # rot = (12,45,70,95,120,150,187,201,230,275,0)
        rot = (45,95,120,170,187,230,250,275,0)
            
        Res = np.zeros((len(rot),velImg,velImg))
        k=0
        for r in rot:
            if r>0:
                params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 
            else:
                params = (velImg,  velImg,velImg,  r,r,  0,0,0,0,   1.0,1.0,  1.0) 
            # r=0
            
            # params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 
            # params = (velImg,  velImg,velImg,  r,r,  -20,20,-20,20,   1.0,1.0,  1.0) 
    
            
            augm_params=[]
            augm_params.append({'Output_size': params[0],
                                'Crop_size': random.randint(params[1],params[2]),
                                'Angle': random.randint(params[3],params[4]),
                                'Transl': (0,0),
                                'Scale': random.uniform(params[9],params[10]),
                                'Flip': False
                                })
            
            augm_params[0]['Angle'] = r
            augm_params[0]['Transl'] = (0,0)
            img = Util.augmentation2(imgO, augm_params) 
            
            augm_params[0]['Angle'] = 0
            if not r==0:
                augm_params[0]['Transl'] = (random.randint(params[5],params[6]),random.randint(params[7],params[8]))
            else:
                augm_params[0]['Transl'] =  (0,0)
            img = Util.augmentation2(img, augm_params)
            # plt.figure()
            # plt.imshow(img[0,:,:])
            # plt.show()
            
            Imgs[0,0,:,:] = img
                
            # sc = 
            with torch.no_grad(): 
                res = net( Imgs.cuda() )
                # res = torch.softmax(res,dim=1)
                res = torch.sigmoid(res)      # for V9_   
            
            augm_params[0]['Angle'] = 0
            augm_params[0]['Transl'] = (-augm_params[0]['Transl'][0],-augm_params[0]['Transl'][1])
            res = Util.augmentation2(res, augm_params) 
            
            augm_params[0]['Angle'] = -r
            augm_params[0]['Transl'] = (0,0)
            res = Util.augmentation2(res, augm_params) 
            
            # res = (res[0,0,:,:].detach().cpu().numpy()>0.5)
            # # .astype(np.dtype('float')),0) 
            # labelled = measure.label(res)
            # rp = measure.regionprops(labelled)
            # if not not rp:
            #     size = max([i.area for i in rp])
            #     resM = morphology.remove_small_objects(res, min_size=size-1)
            # else:
            #     resM=res
            
            Res[k,:,:] = res[0,0,:,:].detach().cpu().numpy()
            k=k+1
            
            # plt.figure
            # plt.imshow( Res[k-1,:,:] )    
            # plt.show()
            
        torch.cuda.empty_cache()
        
        mask = Util.augmentation2(mask, augm_params)
        mask = mask[0,:,:]>0.5
         
        tresh = 0.50
        # res = np.median((Res>0.5).astype(np.dtype('float')),0) 
        # res = np.mean(Res,0) >0.5 
        res2 = np.median(Res,0) > tresh
        # res = np.sum(Res,0) >0.5  
        # res = Res[8,:,:] >0.5
        # res = Res[8,:,:]>tresh
        res1 = Res[8,:,:]>tresh
    
        
        # res=res>0.5
        # save_name = 'Clin_pp' + '_' + str(tresh)
        # plt.figure
        # plt.imshow( res )
        # plt.show()
        
        
        # res = morphology.binary_erosion(res)
        labelled = measure.label(res1)
        rp = measure.regionprops(labelled)
        if not not rp:
            size = max([i.area for i in rp])
            res1M = morphology.remove_small_objects(res1, min_size=size-1)
        else:
            res1M=res1
            
        labelled = measure.label(res2)
        rp = measure.regionprops(labelled)
        if not not rp:
            size = max([i.area for i in rp])
            res2M = morphology.remove_small_objects(res2, min_size=size-1)
        else:
            res2M=res2
            
        res1 = torch.tensor(res1)
        res2 = torch.tensor(res2)
        res1M = torch.tensor(res1M)
        res2M = torch.tensor(res2M)
        # resM = resM[0,:,:].cpu()
        # res = res[0,:,:,].cpu()
        
        # plt.figure()
        # plt.imshow(resM)
        # plt.show()
            
        dice1 = Util.dice_coef( res1, mask ) 
        dice2 = Util.dice_coef( res2, mask )   
        dice1M = Util.dice_coef( res1M, mask )  
        dice2M = Util.dice_coef( res2M, mask )        
     
        A = res1.detach().cpu().numpy()>tresh
        B = mask.detach().cpu().numpy()>0.5
        HD1 = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
                  
        A = res1M.detach().cpu().numpy()>tresh
        B = mask.detach().cpu().numpy()>0.5
        HD1m = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
        
        A = res2.detach().cpu().numpy()>tresh
        B = mask.detach().cpu().numpy()>0.5
        HD2 = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
                  
        A = res2M.detach().cpu().numpy()>tresh
        B = mask.detach().cpu().numpy()>0.5
        HD2m = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))
        
       
        # dif = torch.sum(  (res ^ resM)  )
        # print(dif.item())
        
        res_table.loc[(num,'Name')] =   nPat 
        res_table.loc[(num,'Set')] =   data_list_test.iloc[num]['Set']
        res_table.loc[(num,'Slice')] =   current_index 
        res_table.loc[(num,'Dataset')] =   data_list_test.iloc[num]['Dataset']
        res_table.loc[(num,'Seq')] =   data_list_test.iloc[num]['Series']
        res_table.loc[(num,'HD1')] =   HD1
        res_table.loc[(num,'Dice1')] =   dice1.item()
        res_table.loc[(num,'HD1m')] =   HD1m
        res_table.loc[(num,'Dice1m')] =   dice1M.item()
        res_table.loc[(num,'HD2')] =   HD2
        res_table.loc[(num,'Dice2')] =   dice2.item()
        res_table.loc[(num,'HD2m')] =   HD2m
        res_table.loc[(num,'Dice2m')] =   dice2M.item()
        # res_table.loc[(num,'FP')] =   dif.item()
        
        print(nPat)
        
        # # Fig = plt.figure()
        # # plt.imshow(img[0,:,:], cmap='gray')
        
        # for ii in range(0,2):  
        #     if ii==0:
        #         res = Res[8,:,:]>tresh
        #     else:
        #         res = np.median(Res,0) > tresh  
        
        
        # # res = np.median(Res,0) > tresh
        # res = res2M.detach().cpu().numpy()
        # # # res = Res[8,:,:]>tresh
     
        # resB = res.astype(np.dtype('uint8'))
        # dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
        # ctr = dimg - resB
        
        # GT = (mask.detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
        # dimg = cv2.dilate(GT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
        # ctrGT = dimg - GT
        
        
        # imgB = img[0,:,:].numpy().astype(np.dtype('float'))
        # imgB = (( imgB - imgB.min() ) / (imgB.max() - imgB.min()) )
        # # imgB = ( imgB - 0 ) / (0.7 - 0 )    # for Clinical T1
        # imgB[imgB<0]=0
        # imgB[imgB>1]=1
        # RGB_imgB = cv2.cvtColor((imgB*255).astype(np.dtype('uint8')),cv2.COLOR_GRAY2RGB)
        
        # comp = RGB_imgB[:,:,0]
        # comp[ctr==1]=255
        # comp[ctrGT==1]=0
        # RGB_imgB[:,:,0] = comp
        
        # comp = RGB_imgB[:,:,1]
        # comp[ctr==1]=0
        # comp[ctrGT==1]=255
        # RGB_imgB[:,:,1] = comp
        
        # comp = RGB_imgB[:,:,2]
        # comp[ctr==1]=0
        # comp[ctrGT==1]=0
        # RGB_imgB[:,:,2] = comp
            
        # # Fig = plt.figure()
        # # plt.imshow(imgB,cmap='gray')
        # # plt.show()
        # # plt.draw()
        
        # Fig = plt.figure()
        # plt.imshow(RGB_imgB)
        # plt.text(-10,-10,img_path)
        # plt.show()
        # plt.draw()
        
        # # # Fig = plt.figure()
        # # # plt.imshow( np.mean(Res,0))
        # # # plt.show()
        # # # plt.draw()
    
            
        # Fig.savefig( path_save + '/' + 'res_' + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + str(current_index) + '.png', dpi=150)
        # # Fig.savefig( path_save + '/' + 'res_' + "%.4f" % (dice.item()) + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + str(current_index) + '.png', dpi=150)
        # plt.close(Fig)
        
        # mdic = {"segm_mask": res, "dcm_data": img[0,:,:].numpy(), "prob_maps": Res, "GT":B}
        # savemat(  path_save_data + '/' + "%05d" % (num) + '_' +  nPat + '_' +  str(current_index) + '.mat', mdic)
    
    Util.save_to_excel(res_table, path_save + '/' , 'Res' + save_name + '_' + version)
    
    
