#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:06:46 2022

@author: rj21
"""

#for training version 4, new data, from nifti

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
import pickle
import pydicom as dcm
from scipy.stats import norm
from operator import itemgetter
import pandas as pd
from sklearn.utils import shuffle


import Utilities as Util
import Loaders
# import Network as Network
import Network_Att as Network


lr         = 0.002
L2         = 0.000001
batch      = 8
step_size  = [200,270,300]
sigma      = 0.7
lambda_Cons = 0.001
lambda_Spec = 1.0
num_ite    = 50
num_epch   = step_size[-1]


batchTr = int(np.round(batch))
# step_size = int(np.round(step_size))
num_ite = int(np.round(num_ite))
 
torch.cuda.empty_cache()  
 
# ## Create new netowrk UNet
# net = Network.AttU_Net(img_ch=1,output_ch=1)
# # net = Network.U_Net(img_ch=1,output_ch=1)
# Network.init_weights(net,init_type= 'xavier', gain=0.02)

## Load pretrained model
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v10_0_0.pt")    ## for Unet net
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v10_6_0.pt")   ## for Attention net

version_new = "v12_1_1"

path_data = '/data/rj21/MyoSeg/params_net' + os.sep + 'scans_list_Task744_MyoSeg.xlsx'

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=L2)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

data_list = pd.read_excel(path_data)
data_list_1_cons = pd.read_excel('/data/rj21/MyoSeg/params_net' + os.sep + 'scans_list_unLab_MyoSeg.xlsx')


data_list_1_train = data_list.iloc[data_list.index[data_list['Set'] == "train"].tolist()]
data_list_1_test = data_list.iloc[data_list.index[data_list['Set'] == "test"].tolist()]
# data_list_1_cons = data_list.iloc[data_list.index[data_list_cons['Set'] == "Cons"].tolist()]

data_list_1_train = data_list_1_train.reset_index()
data_list_1_test = data_list_1_test.reset_index()


ind1=[]; ind2=[]; ind3=[]; ind4=[]
diceTr=[]; diceTr_Cons=[]; diceTe=[];
HD_Te=[]; HD_Tr=[]; bestModel=0


D1 = np.zeros((len(data_list_1_train),2))
D1[:,0] = np.arange(0,len(data_list_1_train))
   

for epch in range(0,num_epch):
    mu1, sigma1 = len(data_list_1_train)/10 , sigma*len(data_list_1_train)
    
    net.train(mode=True)
    
    # if epch>10:
    #     sigma = 0.7

    diceTr1=[];  diceTr3=[]; diceTe1=[];  HD1=[]; HD2=[]
               
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for n_ite in range(0,num_ite):
      
        params = (256,  186,276,  -170,170,  -60,60,-60,60,  0.8,1.2,  1.0)

        ## Pro specific datatset 1
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx_Orig = D1[Indx_Sort,0].astype('int')
        # sub_set = list(map(data_list_1_train.__getitem__, Indx_Orig))
        sub_set = data_list_1_train.iloc[Indx_Orig]
        
        loss_Spec, res, Imgs5, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
        # loss_train, res, Imgs, Masks = Network.Training.straightForwardFour(sub_set, net, params, TrainMode=True, Contrast=False)
                                  
        dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTr1.append( np.mean( dice.detach().cpu().numpy() ) )
        # Inds1.append(Indx)
        D1[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
        for b in range(0,batchTr):
            A = res[b,0,:,:].detach().cpu().numpy()>0.5
            B = Masks[b,0,:,:].detach().cpu().numpy()>0.5
            HD1.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
 
        del Masks, res
         
        D1 = D1[D1[:, 1].argsort()]        
        
    ## Consistency regularization
        params = (256,  186,276 ,  -170,170,  -60,60,-60,60 ,  0.8,1.2 , 1.0)
        batchCons = 6
        Indx = np.random.randint(0,len(data_list_1_cons),(batchCons)).tolist()
        sub_set = data_list_1_train.iloc[Indx_Orig]
        loss_cons, _, _, _ = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
        diceTr3.append(1 - loss_cons.detach().cpu().numpy())
       
        
    ## backF - training
        net.train(mode=True)
        if epch>0:
            loss = lambda_Spec*loss_Spec + np.mean(HD1)
            # loss = lambda_Spec*loss_Spec + np.mean(HD1) + lambda_Cons*loss_cons
            # loss = lambda_Other*loss_Other + lambda_Spec*loss_Spec + np.mean(HD1) 
            # loss = lambda_Other*loss_Other + np.mean(HD2) + lambda_Cons*loss_cons 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    pd = norm(mu1,sigma1)
    plt.figure()
    plt.plot(D1[:,1])
    y = pd.pdf([np.linspace(0,np.size(D1,0),np.size(D1,0))]).T
    plt.plot(y/y.max())
    plt.ylim([0.0, 1.1])
    plt.show()
    
    if epch>0:
        scheduler.step()
        
    net.train(mode=False)
   
    ### validation
    params = (256,  256,256,  -0,0,  -0,0,-0,0,    1.0,1.0,   1.0)
    # params = (128,  108,148, -170,170,  -10,10,-10,10)
    batchTe = 32
    data_list_1_test = shuffle(data_list_1_test)

    for num in range(0, int(np.floor(len(data_list_1_test)) *1.0 ) , batchTe ):   
        sub_set = data_list_1_test.iloc[num:num+batchTe]
        with torch.no_grad():
            _, resTe, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        dice = Util.dice_coef( resTe[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() ) 
        diceTe1.append(dice.detach().cpu().numpy()) 
        
        HD2 = []
        for b in range(0,resTe.shape[0]):
            A = resTe[b,0,:,:].detach().cpu().numpy()>0.5
            B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
            HD2.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
    
    if np.nanmean(diceTe1) > bestModel:
        torch.save(net, 'Models/net_' + version_new + '.pt')
        bestModel = np.nanmean(diceTe1)
    
    print(bestModel)
    torch.cuda.empty_cache()
     

    diceTr.append(np.nanmean(diceTr1))
    diceTe.append(np.nanmean(diceTe1))
    diceTr_Cons.append(np.nanmean(diceTr3))

    HD_Tr.append(np.nanmean(HD1))
    HD_Te.append(np.nanmean(HD2))

    
    plt.figure
    plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(resTe[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()

       
    plt.figure()
    plt.plot(diceTr,label='Spec Train')
    plt.plot(diceTe,label='Spec Test')
    plt.plot(diceTr_Cons,label='Consistency StT UnLab')
  
    # plt.ylim([0.6, 0.9])
    plt.legend()
    plt.show()    


file_name = 'Models/net_' + version_new + '_info' + ".pkl"
open_file = open(file_name, "wb")
pickle.dump([diceTr, diceTe, diceTr_Cons, HD_Tr, HD_Te], open_file)
open_file.close()


torch.save(net, 'Models/net_' + version_new + '_end' + '.pt')