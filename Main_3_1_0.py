#for training version 3, ACDC, StT new

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

import Utilities as Util
# import Loaders
import Network

# 
# net = Network.Net(enc_chs=(1,16,32,64,128,256), dec_chs=(256,128,64,32,16), out_sz=(128,128), retain_dim=False, num_class=2)
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_1_5.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v2_1.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=0.00001)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, verbose=True)

## StT LABELLED - JOINT
path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
data_list = Util.CreateDataset_StT_dcm(os.path.normpath( path_data ))

b = int(len(data_list)*0.7)
data_list_1_train = data_list[1:b]
data_list_1_test = data_list[b+1:-1]

## StT LABELLED - JOINT
path_data = '/data/rj21/Data/Data_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
data_list = Util.CreateDataset_StT_dcm(os.path.normpath( path_data ))

# b = int(len(data_list)*0.7)
# data_list_4_train = data_list[1:b]
# data_list_4_test = data_list[b+1:-1]
data_list_4_train = data_list

## ACDC
path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
data_list_2_train, data_list_2_test = Util.CreateDataset(os.path.normpath( path_data ))

## StT UNLABELLED
path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
data_list = Util.CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ))

# b = int(len(data_list)*0.7)
# data_list_3_test = data_list[b+1:-1]
# data_list_3_train = data_list[1:b]
data_list_3_train = data_list


diceTr_Joint=[]
diceTr_ACDC=[]
diceTr_cons=[]
diceTe_Joint=[]
diceTe_ACDC=[]


for epch in range(0,150):
    random.shuffle(data_list_1_train)
    random.shuffle(data_list_2_train)
    random.shuffle(data_list_3_train)
    net.train(mode=True)
    batch = 8
    diceTr1=[]
    diceTr2=[]
    diceTr3=[]
    diceTe1=[]
    diceTe2=[]
        
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for num_ite in range(0,160):
      
    ### Pro StT our dataset JOINT
        # sub_set = data_list_1_train[num:num+batch]
        Indx = np.random.randint(0,len(data_list_1_train),(batch,)).tolist()
        sub_set =list(map(data_list_1_train.__getitem__, Indx))
        
        params = (128,  80,120,  -170,170,  -10,10,-10,10)
        loss_Joint, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True)
                                                   
        # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
        dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
        diceTr1.append(dice.detach().cpu().numpy())
    
    ### Pro ACDC dataset
        # Indx = np.random.randint(0,len(data_list_2_train),(batch,)).tolist()
        # sub_set =list(map(data_list_2_train.__getitem__, Indx))
        
        # # with torch.no_grad(): 
        # params = (128,  100,120,  -170,170,  -10,10,-10,10)
        # loss_ACDC, res2, Imgs2, Masks2 = Network.Training.straightForward(sub_set, net, params, TrainMode=True)
                                                       
        # # train_loss_ACDC.append(loss_ACDC.detach().cpu().numpy())
        # dice = Util.dice_coef( res2[:,0,:,:]>0.5, Masks2[:,0,:,:].cuda() )                
        # diceTr2.append(dice.detach().cpu().numpy())
    
    
    ## Consistency regularization
        Indx = np.random.randint(0,len(data_list_3_train),(batch,)).tolist()
        sub_set = list(map(data_list_3_train.__getitem__, Indx))
        loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True)
        diceTr3.append(1 - loss_cons.detach().cpu().numpy())
        
    ## backF - training
        if epch>0:
            # loss = loss_Joint + 0.0001*loss_ACDC + 0.01*loss_cons
            # loss = loss_Joint
            # loss = loss_Joint + 0.05*loss_ACDC
            loss = loss_Joint + 0.1*loss_cons
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()

        torch.cuda.empty_cache()
    scheduler.step()
    
    net.train(mode=False)
    batch = 512
    random.shuffle(data_list_1_test)
    for num in range(0,len(data_list_1_test)-batch-1, batch):
        sub_set1 = data_list_1_test[num:num+batch]
        params = (128,  80,120,  -180,180,  -10,0,-10,0)
        with torch.no_grad(): 
            _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set1, net, params, TrainMode=False)       
                         
        dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        diceTe1.append(dice.detach().cpu().numpy())
    batch = 100
    # random.shuffle(data_list_2_test)
    # for num in range(0,len(data_list_2_test)-batch-1, batch):    
    #     sub_set2 = data_list_2_test[num:num+batch]   
    #     with torch.no_grad(): 
    #         _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set2, net, params, TrainMode=False)       
                         
    #     dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
    #     diceTe2.append(dice.detach().cpu().numpy())
        
    torch.cuda.empty_cache()
        
    
    diceTr_Joint.append(np.mean(diceTr1))
    diceTr_ACDC.append(np.mean(diceTr2))
    diceTr_cons.append(np.mean(diceTr3))
    diceTe_Joint.append(np.mean(diceTe1))
    diceTe_ACDC.append(np.mean(diceTe2))


    # print(np.mean(diceTe))
    
    plt.figure
    plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(resTE[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()
    
    # plt.figure
    # plt.imshow(Imgs_P[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(res[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    
    # plt.figure
    # plt.imshow(Imgs_P[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(res_P[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    
    # plt.figure
    # plt.imshow(res[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.imshow(res_P[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    
    # plt.figure
    # plt.plot(train_loss_Joint)
    # plt.plot(train_loss_ACDC)
    # plt.ylim([0.0, 1.2])
    # plt.show()
    
    plt.figure()
    plt.plot(diceTr_Joint,label='StT Labeled TR')
    plt.plot(diceTe_Joint,label='StT Labeled TE')
    plt.plot(diceTr_ACDC,label='ACDC TR')
    plt.plot(diceTe_ACDC,label='ACDC TE')
    plt.plot(diceTr_cons,label='Consistency StT unL')
  
    plt.ylim([0.0, 0.9])
    plt.legend()
    plt.show()
    
    # plt.figure
    # plt.plot(diceTr_cons)
    # plt.show()
    
version = "v3_1_9_6"

torch.save(net, 'Models/net_' + version + '.pt')

file_name = "Models/Res_net_" + version + ".pkl"
open_file = open(file_name, "wb")
pickle.dump([diceTr_Joint,diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC], open_file)
open_file.close()

# open_file = open(file_name, "rb")
# res = pickle.load(open_file)
# open_file.close()

# open_file = open(file_name, "rb")
# diceTr_Joint,diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC = pickle.load(open_file)
# open_file.close()