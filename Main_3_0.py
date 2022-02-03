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


net = Network.Net(enc_chs=(1,16,32,64,128,256), dec_chs=(256,128,64,32,16), out_sz=(128,128), retain_dim=False, num_class=2)
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_4.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v2_1.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00024, weight_decay=0.000001)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, verbose=True)

# ## StT LABELLED
# path_data = '/data/rj21/Data/Data_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
# data_list_test = Util.CreateDataset_StT_dcm(os.path.normpath( path_data ))
# b = int(len(data_list_test)*0.7)
# data_list_1_train = data_list_test[1:b]
# data_list_1_test = data_list_test[b+1:-1]

## ACDC
path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
data_list_2_train, data_list_2_test = Util.CreateDataset(os.path.normpath( path_data ))

# ## StT UNLABELLED
# path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
# data_list_3_train = Util.CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ))
# b = int(len(data_list_3_train)*0.7)
# data_list_3_test = data_list_3_train[b+1:-1]
# data_list_3_train = data_list_3_train[1:b]


diceTr_Joint=[]
diceTr_ACDC=[]
diceTr_cons=[]
# train_loss_Joint=[]
# train_loss_ACDC=[]

for epch in range(0,80):
    # random.shuffle(data_list_1_train)
    random.shuffle(data_list_2_train)
    # random.shuffle(data_list_3_train)
    net.train(mode=True)
    batch = 16
    diceTr1=[]
    diceTr2=[]
    diceTr3=[]
        
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for num_ite in range(0,40):
      
    # ### Pro StT our dataset JOINT
    #     # sub_set = data_list_1_train[num:num+batch]
    #     Indx = np.random.randint(0,len(data_list_1_train),(batch,)).tolist()
    #     sub_set =list(map(data_list_1_train.__getitem__, Indx))
        
    #     params = (128,  100,120,  -170,170,  -10,10,-10,10)
    #     loss_Joint, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True)
                                                   
    #     # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
    #     dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
    #     diceTr1.append(dice.detach().cpu().numpy())
    
    ### Pro ACDC dataset
        Indx = np.random.randint(0,len(data_list_2_train),(batch,)).tolist()
        sub_set =list(map(data_list_2_train.__getitem__, Indx))
        
        # with torch.no_grad(): 
        params = (128,  100,120,  -170,170,  -10,10,-10,10)
        loss_ACDC, res2, Imgs2, Masks2 = Network.Training.straightForward(sub_set, net, params, TrainMode=True)
                                                       
        # train_loss_ACDC.append(loss_ACDC.detach().cpu().numpy())
        dice = Util.dice_coef( res2[:,0,:,:]>0.5, Masks2[:,0,:,:].cuda() )                
        diceTr2.append(dice.detach().cpu().numpy())
    
    
    # ## Consistency regularizzation
    #     Indx = np.random.randint(0,len(data_list_3_train),(batch,)).tolist()
    #     sub_set = list(map(data_list_3_train.__getitem__, Indx))
    #     loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True)
    #     diceTr3.append(1 - loss_cons.detach().cpu().numpy())
        
    ## backF - training
        # loss = loss_Joint + 0.05*loss_ACDC + 0.01*loss_cons
        loss = loss_ACDC
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        optimizer.step()

        torch.cuda.empty_cache()
    scheduler.step()
    
    
    # batch = 200
    # net.train(mode=False)
    # random.shuffle(data_list_test)

    # for num in range(0,len(data_list_test)-batch-1, batch):
    #     sub_set = data_list_test[num:num+batch]
        
    #     with torch.no_grad(): 
    #         params = (128,  100,120,  -0,0,  -0,0,-0,0)
    #         loss, res, Imgs, Masks = Network.Training.straightForward(sub_set, net, params, batch, TrainMode=False)
                 
                         
    #     dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
    #     diceTe.append(dice.detach().cpu().numpy())
        
    #     torch.cuda.empty_cache()
    
    diceTr_Joint.append(np.mean(diceTr1))
    diceTr_ACDC.append(np.mean(diceTr2))
    diceTr_cons.append(np.mean(diceTr3))

    # print(np.mean(diceTe))
    
    # plt.figure
    # plt.imshow(Imgs1[2,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(res1[2,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    
    plt.figure
    plt.imshow(Imgs2[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(Masks2[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()
    
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
    
    plt.figure
    # plt.plot(diceTr_Joint)
    plt.plot(diceTr_ACDC)
    # plt.plot(diceTr_cons)
    plt.show()
    
    # plt.figure
    # plt.plot(diceTr_cons)
    # plt.show()

torch.save(net, 'Models/net_v3_0_0.pt')