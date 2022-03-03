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
from scipy.stats import norm

import Utilities as Util
# import Loaders
import Network

# 
# net = Network.Net(enc_chs=(1,32,64,128,256), dec_chs=(256,128,64,32), out_sz=(128,128), head=(128), retain_dim=False, num_class=2)
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v7_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v5_0_6.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00000)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

## ACDC
path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
data_list_2_train, data_list_2_test = Util.CreateDataset(os.path.normpath( path_data ))


# ## StT LABELLED - JOINT
path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
data_list = Util.CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
b = int(len(data_list)*0.8)
data_list_1_train = data_list[1:b]
# data_list_1_test = data_list[b+1:-1]
# data_list_1_train = data_list
# data_list_1_test = data_list

## StT LABELLED - P1-30
path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
data_list = Util.CreateDataset_StT_P_dcm(os.path.normpath( path_data ))
b = int(len(data_list)*0.55)
data_list_4_train = data_list[1:b]
data_list_4_test = data_list[b+1:-1]
# data_list_4_test = data_list
# data_list_4_test = data_list[b+1:-1]
# random.shuffle(data_list_4_train)

## Dataset - MyoPS
path_data = '/data/rj21/Data/Data_MyoPS'  # Linux bioeng358
data_list = Util.CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
# b = int(len(data_list)*0.75)
# data_list_5_train = data_list[0:b]
# data_list_5_test = data_list[b+1:-1]
data_list_5_train = data_list
# random.shuffle(data_list_4_train)

## Dataset - EMIDEC
path_data = '/data/rj21/Data/Data_emidec'  # Linux bioeng358
data_list = Util.CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
# b = int(len(data_list)*0.75)
# data_list_6_train = data_list[0:b]
# data_list_6_test = data_list[b+1:-1]
data_list_6_train = data_list

# StT UNLABELLED
path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
data_list = Util.CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ))
# b = int(len(data_list)*0.7)
# data_list_3_test = data_list[b+1:-1]
# data_list_3_train = data_list[1:b]
data_list_3_train = data_list
random.shuffle(data_list_3_train)


diceTr_Joint=[]
diceTr_Clin=[]
diceTr_ACDC=[]
diceTr_cons=[]
diceTr_MyoPS=[]
diceTr_Emidec=[]
diceTe_Joint=[]
diceTe_ACDC=[]
diceTe_StT=[]
diceTe_Clin=[]
diceTe_MyoPS=[]
diceTe_Emidec=[]
HD_Te_Joint=[]
HD_Te_Clin=[]


# num_iter = 60
batchTr = 8
D1 = np.zeros((len(data_list_1_train),2))
D1[:,0] = np.arange(0,len(data_list_1_train))
D2 = np.zeros((len(data_list_2_train),2))
D2[:,0] = np.arange(0,len(data_list_2_train))
D4 = np.zeros((len(data_list_4_train),2))
D4[:,0] = np.arange(0,len(data_list_4_train))
D5 = np.zeros((len(data_list_5_train),2))
D5[:,0] = np.arange(0,len(data_list_5_train))
D6 = np.zeros((len(data_list_6_train),2))
D6[:,0] = np.arange(0,len(data_list_6_train))

mu1, sigma1 = len(data_list_1_train)/10 ,  0.70*len(data_list_1_train)
mu2, sigma2 = len(data_list_2_train)/10 ,  0.70*len(data_list_2_train)
mu4, sigma4 = len(data_list_4_train)/10 ,  0.70*len(data_list_4_train)
mu5, sigma5 = len(data_list_5_train)/10 ,  0.70*len(data_list_5_train)
mu6, sigma6 = len(data_list_6_train)/10 ,  0.70*len(data_list_6_train)

for epch in range(0,122):
    # random.shuffle(data_list_1_train)
    # # random.shuffle(data_list_2_train)
    # random.shuffle(data_list_3_train)
    # random.shuffle(data_list_4_train)
    net.train(mode=True)
    diceTr1=[]
    diceTr2=[]
    diceTr3=[]
    diceTr4=[]
    diceTr5=[]
    diceTr6=[]
    diceTe1=[]
    diceTe2=[]
    diceTe4=[]
    diceTe5=[]
    diceTe6=[]
    HD4=[]
    HD1=[]
    
    Inds1=[]
    Inds2=[]
    Inds4=[]
    Inds5=[]
    Inds6=[]
    
        
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for num_ite in range(0,50):
      
    ## Pro StT our dataset JOINT
        # sub_set = data_list_1_train[num:num+batch]
        Indx = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx = D1[Indx,0].astype('int')
        # Indx = np.random.randint(0,len(data_list_1_train),(batchTr,)).tolist()
        sub_set = list(map(data_list_1_train.__getitem__, Indx))
        
        params = (128,  100,120,  -170,170,  -10,10,-10,10)
        loss_Joint, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
        dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
        diceTr1.append(dice.detach().cpu().numpy())
        Inds1.append(Indx)
        
    ### Pro StT our dataset CLINICAL
        # sub_set = data_list_4_train[num:num+batch]
        Indx = Util.rand_norm_distrb(batchTr, mu4, sigma4, [0,len(data_list_4_train)]).astype('int')
        Indx = D4[Indx,0].astype('int')
        # Indx = np.random.randint(0,len(data_list_4_train),(batchTr,)).tolist()
        sub_set =list(map(data_list_4_train.__getitem__, Indx))
        
        # params = (128,  80,120,  -170,170,  -20,20,-20,20)
        loss_Clin, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
        dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
        diceTr4.append(dice.detach().cpu().numpy())
        Inds4.append(Indx)
        
    ### Pro MyoPS datatse
        Indx = Util.rand_norm_distrb(batchTr, mu5, sigma5, [0,len(data_list_5_train)]).astype('int')
        Indx = D5[Indx,0].astype('int')
        # Indx = np.random.randint(0,len(data_list_4_train),(batchTr,)).tolist()
        sub_set =list(map(data_list_5_train.__getitem__, Indx))
        
        # params = (128,  80,128,  -170,170,  -10,10,-10,10)
        loss_MyoPS, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
        dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
        diceTr5.append(dice.detach().cpu().numpy())
        Inds5.append(Indx)    
        
    ## Pro Emidec datatse
        Indx = Util.rand_norm_distrb(batchTr, mu6, sigma6, [0,len(data_list_6_train)]).astype('int')
        Indx = D6[Indx,0].astype('int')
        # Indx = np.random.randint(0,len(data_list_4_train),(batchTr,)).tolist()
        sub_set =list(map(data_list_6_train.__getitem__, Indx))
        
        # params = (128,  80,128,  -170,170,  -10,10,-10,10)
        loss_Emidec, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        # train_loss_Joint.append(loss_Joint.detach().cpu().numpy())
        dice = Util.dice_coef( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
        diceTr6.append(dice.detach().cpu().numpy())
        Inds6.append(Indx) 
        
    # ## Pro ACDC dataset
    #     Indx = Util.rand_norm_distrb(batchTr, mu2, sigma2, [0,len(data_list_2_train)]).astype('int')
    #     Indx = D2[Indx,0].astype('int')
    #     sub_set =list(map(data_list_2_train.__getitem__, Indx))
    #     Inds2.append(Indx)  
        
    #     # params = (128,  80,120,  -170,170,  -20,20,-20,20)
    #     loss_ACDC, res2, Imgs2, Masks2 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
    #     dice = Util.dice_coef( res2[:,0,:,:]>0.5, Masks2[:,0,:,:].cuda() )                
    #     diceTr2.append(dice.detach().cpu().numpy())
    
    
    ## Consistency regularization
        # params = (128,  80,120,  -170,170,  -10,10,-10,10)
        Indx = np.random.randint(0,len(data_list_3_train),(batchTr,)).tolist()
        sub_set = list(map(data_list_3_train.__getitem__, Indx))
        loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
        diceTr3.append(1 - loss_cons.detach().cpu().numpy())
        
    ## backF - training
        net.train(mode=True)
        if epch>0:
            # loss = loss_Joint + 0.1*loss_cons
            # loss = loss_ACDC + 0.01*loss_cons
            # loss = 0.5*loss_Joint  + loss_Clin + 0.01*loss_cons
            # loss = 0.3*loss_Joint  + 0.5*loss_Clin + 0.2*loss_MyoPS + 0.01*loss_cons
            loss = 0.2*loss_Joint + 0.5*loss_Clin + 0.2*loss_MyoPS + 0.1*loss_Emidec  + 0.01*loss_cons
            # loss = loss_ACDC + 0.01*loss_cons
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
            optimizer.step()


    if epch>0:
        scheduler.step()
        
    net.train(mode=False)
    
    # ## Joint
    # params = (128,  80,120,  -0,0,  0,0,0,0)
    # batch = 56
    # random.shuffle(data_list_1_test)
    # # for num in range(0,len(data_list_1_test)-batch-1, batch):   
    # for num in range(0,1): 
    #     sub_set2 = data_list_1_test[num:num+batch]   
    #     with torch.no_grad(): 
    #         _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set2, net, params, TrainMode=False, Contrast=False)       
                         
    #     dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
    #     diceTe1.append(dice.detach().cpu().numpy())
        
    ### StT lab
    params = (128,  100,120,  -0,0,  0,0,0,0)
    batch = 128
    random.shuffle(data_list_4_test)
    # for num in range(0,len(data_list_4_test), batch):
    for num in range(0,4):   
        sub_set = data_list_4_test[num:num+batch]
        with torch.no_grad():
            _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        # diceTe1.append(dice.detach().cpu().numpy())
        diceTe4.append(dice.detach().cpu().numpy())
         
        for b in range(0,batch):
            A = resTE[b,0,:,:].detach().cpu().numpy()>0.5
            B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
            HD4.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
      
    ### StT Joint
    # params = (128,  80,120,  -0,0,  0,0,0,0)
    # batch = 256
    # random.shuffle(data_list_1_test)
    # # for num in range(0,len(data_list_4_test), batch):
    # for num in range(0,3):   
    #     sub_set = data_list_1_test[num:num+batch]
    #     with torch.no_grad():
    #         _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
    #     dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
    #     diceTe1.append(dice.detach().cpu().numpy())
         
    #     for b in range(0,batch):
    #         A = resTE[b,0,:,:].detach().cpu().numpy()>0.5
    #         B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
    #         HD1.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))

    # ### ACDC     
    # params = (128,  80,120,  -0,0,  0,0,0,0)
    # batch = 128
    # random.shuffle(data_list_2_test)
    # # for num in range(0,len(data_list_1_test)-batch-1, batch):   
    # for num in range(0,3): 
    #     sub_set = data_list_2_test[num:num+batch]   
    #     with torch.no_grad(): 
    #         _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
    #     dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
    #     diceTe2.append(dice.detach().cpu().numpy())
    
    torch.cuda.empty_cache()
        

    diceTr_Joint.append(np.mean(diceTr1))
    diceTr_ACDC.append(np.mean(diceTr2))
    diceTr_MyoPS.append(np.mean(diceTr5))
    diceTr_Emidec.append(np.mean(diceTr6))
    diceTr_cons.append(np.mean(diceTr3))
    diceTr_Clin.append(np.mean(diceTr4))
    
    # diceTe_Joint.append(np.mean(diceTe1))
    # diceTe_ACDC.append(np.mean(diceTe2))
    # # diceTe_MyoPS.append(np.mean(diceTe5))
    diceTe_Clin.append(np.mean(diceTe4))
    
    # HD_Te_Joint.append(np.nanmean(HD1))
    # HD_Te_Clin.append(np.nanmean(HD4))
    
    # plt.figure()
    # # plt.hist(np.concatenate(Inds1),256)
    # plt.hist((D4[200:700,0]),256)
    # plt.hist((D4[-500:,0]),256)
    # plt.show()
    
    D1 = D1[D1[:, 0].argsort()]
    D1[np.concatenate(np.array(Inds1)),1] = np.concatenate(np.tile(np.array(diceTr1),(batchTr,1)).T)
    D1 = D1[D1[:, 1].argsort()]
    D2 = D2[D2[:, 0].argsort()]
    # D2[np.concatenate(np.array(Inds2)),1] = np.concatenate(np.tile(np.array(diceTr2),(batchTr,1)).T)
    # D2 = D2[D2[:, 1].argsort()]
    D4 = D4[D4[:, 0].argsort()]
    D4[np.concatenate(np.array(Inds4)),1] = np.concatenate(np.tile(np.array(diceTr4),(batchTr,1)).T)
    D4 = D4[D4[:, 1].argsort()]
    D5 = D5[D5[:, 0].argsort()]
    D5[np.concatenate(np.array(Inds5)),1] = np.concatenate(np.tile(np.array(diceTr5),(batchTr,1)).T)
    D5 = D5[D5[:, 1].argsort()]
    D6 = D6[D6[:, 0].argsort()]
    D6[np.concatenate(np.array(Inds6)),1] = np.concatenate(np.tile(np.array(diceTr6),(batchTr,1)).T)
    D6 = D6[D6[:, 1].argsort()]
    
    # pd = norm(mu1,sigma1)
    # y = pd.pdf([np.linspace(0,np.size(D1,0),np.size(D1,0))]).T

    # plt.figure()
    # plt.stem(D1[:,1]*3.5)
    # plt.plot(y/y.max(),'r')
    # plt.ylim([0.0, 1.0])
    # plt.show()
    

    
    pd = norm(mu6,sigma6)
    plt.figure()
    plt.plot(D6[:,1])
    y = pd.pdf([np.linspace(0,np.size(D6,0),np.size(D6,0))]).T
    plt.plot(y/y.max())
    plt.ylim([0.0, 1.1])
    plt.show()
    
    # print(np.mean(diceTe))
    
    plt.figure
    plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(resTE[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.imshow(MasksTE[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
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
    plt.plot(diceTr_Joint,label='Joint Train')
    plt.plot(diceTe_Clin,label='Clinic Test')
    # plt.plot(diceTe_Joint,label='Joint Test')
    plt.plot(diceTr_Clin,label='Clinic Train')
    # plt.plot(diceTr_ACDC,label='ACDC Train')
    # plt.plot(diceTe_ACDC,label='ACDC Test')
    plt.plot(diceTr_MyoPS,label='MyoPS Train')
    plt.plot(diceTr_Emidec,label='Emidec Train')
    # # plt.plot(diceTe_MyoPS,label='MyoPS Test')
    plt.plot(diceTr_cons,label='Consistency StT UnLab')  
  
    plt.ylim([0.0, 0.9])
    plt.legend()
    plt.show()    
    
    # plt.figure()
    # plt.plot(HD_Te_Clin,label='HD Clinical Test') 
    # plt.ylim([0.0, 5])
    # plt.legend()
    # plt.show()
    
    # plt.figure
    # plt.plot(diceTr_cons)
    # plt.show()
    
version = "v7_3_1"
torch.save(net, 'Models/net_' + version + '.pt')

file_name = "Models/Res_net_" + version + ".pkl"
open_file = open(file_name, "wb")
pickle.dump([diceTr_Joint,diceTr_Emidec, diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC,diceTe_StT,HD_Te_Clin], open_file)
# pickle.dump([diceTr_Joint,diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC,diceTe_StT,D4], open_file)
open_file.close()

# version = "v3_2_1"
# open_file = open(file_name, "rb")
# res = pickle.load(open_file)
# open_file.close()

# open_file = open(file_name, "rb")
# diceTr_Joint,diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC,diceTe_StT = pickle.load(open_file)
# open_file.close()