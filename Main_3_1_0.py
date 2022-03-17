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
import Loaders
import Network


lr         = 0.00001
L2         = 0.000001
batch      = 16
step_size  = 5
sigma      = 0.7
lambda_Cons = 0.1
lambda_Other = 1.0
num_ite    = 30
num_epch = 10


batchTr = int(np.round(batch))
step_size = int(np.round(step_size))
num_ite = int(np.round(num_ite))
 
# net = Network.Net(enc_chs=(1,32,64,128,256), dec_chs=(256,128,64,32), out_sz=(128,128), head=(128), retain_dim=False, num_class=2)
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v7_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v8_0_3.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v8_2_2.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=L2)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, verbose=False)


data_list_2_train, data_list_3_train = Loaders.CreateDataset()
# random.shuffle(data_list_2_train)

## StT LABELLED - P1-30
path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ), 'A','')
# b = int(len(data_list)*0.70)
# data_list_1_train = data_list[1:b]
# data_list_1_test = data_list[b+1:-1]
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ), 'P','')
# b = int(len(data_list)*0.70)
# data_list_1_train = data_list_1_train + data_list[1:b]
# data_list_1_test = data_list_1_test +  data_list[b+1:-1]
data_list_2_train = data_list_2_train + data_list

path_data = '/data/rj21/Data/Data_T2_Alina/dcm_resaved'  # Linux bioeng358
data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
b = int(len(data_list)*0.6)
data_list_1_train = data_list[1:b]
data_list_1_test = data_list[b+1:-1]


diceTr_Clin=[]; diceTr_Other=[]; diceTr_Cons=[]; diceTe_Clin=[]; HD_Te_Clin=[]

# num_iter = 60
# batchTr = 24
D1 = np.zeros((len(data_list_1_train),2))
D1[:,0] = np.arange(0,len(data_list_1_train))
D2 = np.zeros((len(data_list_2_train),2))
D2[:,0] = np.arange(0,len(data_list_2_train))
   

for epch in range(0,num_epch):
    mu1, sigma1 = len(data_list_1_train)/10 , sigma*len(data_list_1_train)
    mu2, sigma2 = len(data_list_2_train)/10 ,  sigma*len(data_list_2_train)
    
    net.train(mode=True)
    
    # if epch>10:
    #     sigma = 0.7

    diceTr1=[]; diceTr2=[]; diceTr3=[]; diceTe1=[]; iceTe2=[];  HD1=[]
    Inds1=[]; Inds2=[]; Inds4=[]; Inds5=[];   Inds6=[]
               
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for n_ite in range(0,num_ite):
      
        ## Pro StT our dataset CLINIC
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx_Orig = D1[Indx_Sort,0].astype('int')
        sub_set = list(map(data_list_1_train.__getitem__, Indx_Orig))
        
        params = (128,  60,120,  -170,170,  -20,20,-20,20)
        loss_Clin, res, _, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTr1.append( np.mean( dice.detach().cpu().numpy() ) )
        # Inds1.append(Indx)
        D1[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
        # D1 = D1[D1[:, 0].argsort()]
        
        # ## Pro Other dataset
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu2, sigma2, [0,len(data_list_2_train)]).astype('int')
        Indx_Orig = D2[Indx_Sort,0].astype('int')
        sub_set = list(map(data_list_2_train.__getitem__, Indx_Orig))  
        
        loss_Other, res, _, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
        dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTr2.append(dice.detach().cpu().numpy())
        D2[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())

    
        D1 = D1[D1[:, 1].argsort()]
        D2 = D2[D2[:, 1].argsort()]
    
    ## Consistency regularization
        params = (128,  80,120,  -170,170,  -10,10,-10,10)
        batchCons = 16
        Indx = np.random.randint(0,len(data_list_3_train),(batchCons,)).tolist()
        sub_set = list(map(data_list_3_train.__getitem__, Indx))
        loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
        diceTr3.append(1 - loss_cons.detach().cpu().numpy())
        
    ## backF - training
        net.train(mode=True)
        if epch>0:
            # loss = loss_Clin + lambda_Other*loss_Other
            loss = loss_Clin + lambda_Other*loss_Other + lambda_Cons*loss_cons
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
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
   
    ### StT lab
    params = (128,  90,100,  -0,0,  0,0,0,0)
    batch = 16
    random.shuffle(data_list_1_test)
    # for num in range(0,len(data_list_4_test), batch):
    for num in range(0,5):   
        sub_set = data_list_1_test[num:num+batch]
        with torch.no_grad():
            _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        diceTe1.append(dice.detach().cpu().numpy())
         
        for b in range(0,batch):
            A = resTE[b,0,:,:].detach().cpu().numpy()>0.5
            B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
            HD1.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
    
    torch.cuda.empty_cache() 
     

    diceTr_Other.append(np.mean(diceTr2))
    diceTr_Cons.append(np.mean(diceTr3))
    diceTr_Clin.append(np.mean(diceTr1))  
    diceTe_Clin.append(np.mean(diceTe1))
    HD_Te_Clin.append(np.nanmean(HD1))

    
    # print(np.mean(diceTe))
# for i in range(0,200):
    plt.figure
    plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(resTE[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()
       
    plt.figure()
    plt.plot(diceTr_Clin,label='Clinic Train')
    plt.plot(diceTe_Clin,label='Clinic Test')
    plt.plot(diceTr_Other,label='Other Train')
    plt.plot(diceTr_Cons,label='Consistency StT UnLab')  
  
    plt.ylim([0.0, 0.9])
    plt.legend()
    plt.show()    
    
    # plt.figure()
    # plt.plot(HD_Te_Clin,label='HD Clinical Test') 
    # plt.ylim([0.0, 5])
    # plt.legend()
    # plt.show()
    
version = "v8_2_5"
torch.save(net, 'Models/net_' + version + '.pt')

file_name = "Models/Res_net_" + version + ".pkl"
open_file = open(file_name, "wb")
pickle.dump([diceTr_Clin, diceTe_Clin, diceTr_Other, diceTr_Cons, HD_Te_Clin], open_file)
open_file.close()

# version = "v3_2_1"
# open_file = open(file_name, "rb")
# res = pickle.load(open_file)
# open_file.close()

# open_file = open(file_name, "rb")
# diceTr_Joint,diceTr_ACDC,diceTr_cons,diceTe_Joint,diceTe_ACDC,diceTe_StT = pickle.load(open_file)
# open_file.close()