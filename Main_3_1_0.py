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
import Network as Network
# import Network_v9 as Network


lr         = 0.001
L2         = 0.000001
batch      = 16
step_size  = 10
sigma      = 0.7
lambda_Cons = 0.01
lambda_Other = 0.5
lambda_Train = 1.0
num_ite    = 50
num_epch   = 50


batchTr = int(np.round(batch))
step_size = int(np.round(step_size))
num_ite = int(np.round(num_ite))
 
torch.cuda.empty_cache()   
 
# net = Network.Net(enc_chs=(4,32,64,128,256), dec_chs=(256,128,64,32), out_sz=(128,128), head=(128), retain_dim=False, num_class=2)
# net = Network.AttU_Net(img_ch=1,output_ch=1)
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v9_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v7_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v8_2_3.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v8_2_0.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v8_4_0.pt")

# Network.init_weights(net,init_type= 'xavier', gain=0.02)

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=L2)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, verbose=False)


data_list_2_train, data_list_3_train = Loaders.CreateDataset()
# random.shuffle(data_list_2_train)

data_list_1_train=[];data_list_1_test=[];

## ACDC
# path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
# data_list_1_train, data_list_1_test = Loaders.CreateDataset_ACDC(os.path.normpath( path_data ))
  
# # ## StT LABELLED - P1-30
# path_data = '/data/rj21/Data/Data_1mm/Clin'  # Linux bioeng358
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ), 'A','')
# b = int(len(data_list)*0.70)
# data_list_1_train = data_list_1_train + data_list[1:b]
# data_list_1_test = data_list_1_test + data_list[b+1:-1]
# # data_list_2_train = data_list_2_train + data_list

# ## StT LABELLED - JOINT
# path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
data_list = Loaders.CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
b = int(len(data_list)*0.80)
# data_list_1_train = data_list[1:b]
# data_list_1_test = data_list[b+1:-1]
data_list_1_train = data_list_1_train + data_list[1:b]
data_list_1_test = data_list_1_test + data_list[b+1:-1]
    

# # ## Alinas data
# # path_data = '/data/rj21/Data/Data_1mm/T2_alina'  # Linux bioeng358
# # data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
# # b = int(len(data_list)*0.8)
# # data_list_1_train = data_list_1_train + data_list[1:b]
# # data_list_1_test = data_list_1_test + data_list[b+1:-1]


diceTr_Clin=[]; diceTr_Other=[]; diceTr_Cons=[]; diceTe_Clin=[]; 
HD_Te_Clin=[]; HD_Tr_Clin=[]; MI=[]

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
    Inds1=[]; Inds2=[]; Inds4=[]; Inds5=[];   Inds6=[] ; mi=[]
               
    # for num_ite in range(0,len(data_list_1_train)-batch-1, batch):
    # for num_ite in range(0,len(data_list_1_train)/batch):
    for n_ite in range(0,num_ite):
      
        ## Pro StT our dataset CLINIC
        Indx_Sort = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
        Indx_Orig = D1[Indx_Sort,0].astype('int')
        sub_set = list(map(data_list_1_train.__getitem__, Indx_Orig))
        
        params = (256,  226,286 ,  -170,170,  -40,40,-40,40)
        # params = (128,  108,148, -170,170,  -10,10,-10,10)

        loss_train, res, Imgs, Masks = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                   
        dice = Util.dice_coef_batch( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTr1.append( np.mean( dice.detach().cpu().numpy() ) )
        # Inds1.append(Indx)
        D1[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
        
        # ## Pro Other dataset
        # Indx_Sort = Util.rand_norm_distrb(batchTr, mu2, sigma2, [0,len(data_list_2_train)]).astype('int')
        # Indx_Orig = D2[Indx_Sort,0].astype('int')
        # sub_set = list(map(data_list_2_train.__getitem__, Indx_Orig))  
        
        # loss_Other, res2, _, Masks2 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
        # dice = Util.dice_coef_batch( res2[:,0,:,:]>0.5, Masks2[:,0,:,:].cuda() )                
        # diceTr2.append(dice.detach().cpu().numpy())
        # D2[np.array(Indx_Sort),1] = np.array(dice.detach().cpu().numpy())
   
        D1 = D1[D1[:, 1].argsort()]
        # D2 = D2[D2[:, 1].argsort()]

        HD2=[]
        # for b in range(0,batchTr):
        #     A = res[b,0,:,:].detach().cpu().numpy()>0.5
        #     B = Masks[b,0,:,:].detach().cpu().numpy()>0.5
        #     HD2.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
    
        
    # ## Consistency regularization
    #     params = (256,  120,128,  -170,170,  -10,10,-10,10)
    #     batchCons = 16
    #     Indx = np.random.randint(0,len(data_list_3_train),(batchCons,)).tolist()
    #     sub_set = list(map(data_list_3_train.__getitem__, Indx))
    #     loss_cons, _, _, _ = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
    #     diceTr3.append(1 - loss_cons.detach().cpu().numpy())
        
    ## backF - training
        net.train(mode=True)
        if epch>0:
            loss = lambda_Train*loss_train
            # loss = lambda_Train*loss_train + np.mean(HD2) + lambda_Other*loss_Other + lambda_Cons*loss_cons 
            # loss = lambda_Train*loss_train + lambda_Other*loss_Other
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

    # pd = norm(mu2,sigma2)
    # plt.figure()
    # plt.plot(D2[:,1])
    # y = pd.pdf([np.linspace(0,np.size(D2,0),np.size(D2,0))]).T
    # plt.plot(y/y.max())
    # plt.ylim([0.0, 1.1])
    # plt.show()
    
    if epch>0:
        scheduler.step()
        
    net.train(mode=False)
   
    ### validation
    params = (256,  256,256,  -0,0,  -0,0,-0,0)
    # params = (128,  108,148, -170,170,  -10,10,-10,10)
    batchTe = 64
    random.shuffle(data_list_1_test)
    # for num in range(0,len(data_list_1_test), batchTe):
    for num in range(0,5):   
        sub_set = data_list_1_test[num:num+batchTe]
        with torch.no_grad():
            _, resTe, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                         
        dice = Util.dice_coef( resTe[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
        diceTe1.append(dice.detach().cpu().numpy())
         
        # for b in range(0,batchTe):
        #     A = resTe[b,0,:,:].detach().cpu().numpy()>0.5
        #     B = MasksTe[b,0,:,:].detach().cpu().numpy()>0.5
        #     HD1.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
    
    torch.cuda.empty_cache() 
     

    diceTr_Other.append(np.mean(diceTr2))
    diceTr_Cons.append(np.mean(diceTr3))
    diceTr_Clin.append(np.mean(diceTr1))  
    diceTe_Clin.append(np.mean(diceTe1))
    HD_Te_Clin.append(np.nanmean(HD1))
    HD_Tr_Clin.append(np.nanmean(HD2))

    
    # print(np.mean(diceTe))
# for i in range(0,200):
    
    plt.figure
    plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(resTe[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()
    # plt.figure
    # plt.imshow(ImgsTe[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(resTE[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
        
    # plt.figure()
    # plt.plot(MI,label='mutual information') 
    
    # plt.figure()
    # plt.plot(HD_Tr_Clin,label='HD train') 
    # plt.plot(HD_Te_Clin,label='HD test') 
    # plt.show()
       
    plt.figure()
    plt.plot(diceTr_Clin,label='Joint Train')
    plt.plot(diceTe_Clin,label='Joint Test')
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
    
version = "v8_4_1"
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