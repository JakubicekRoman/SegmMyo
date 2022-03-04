
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random
import pickle
from bayes_opt import BayesianOptimization

import Utilities as Util
import Network


def get_value(**params):
    lr         = params['lr']
    batch      = params['batch']
    step_size  = params['step_size']
    sigma      = params['sigma']
    lamda_cons = params['lambda_cons']
    
    batch = int(np.round(batch))
    step_size = int(np.round(step_size))
    
    
    # net = Network.Net(enc_chs=(1,32,64,128,256), dec_chs=(256,128,64,32), out_sz=(128,128), head=(128), retain_dim=False, num_class=2)
    net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")
    
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.000001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, verbose=False)
    
    
    # ## StT LABELLED - JOINT
    path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
    data_list = Util.CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
    b = int(len(data_list)*0.8)
    data_list_1_train = data_list[1:b]
    
    
    ## StT LABELLED - P1-30
    path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
    data_list = Util.CreateDataset_StT_P_dcm(os.path.normpath( path_data ))
    b = int(len(data_list)*0.60)
    data_list_4_train = data_list[1:b]
    data_list_4_test = data_list[b+1:-1]
    
    
    ## Dataset - MyoPS
    path_data = '/data/rj21/Data/Data_MyoPS'  # Linux bioeng358
    data_list = Util.CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    data_list_5_train = data_list
    
    ## Dataset - EMIDEC
    path_data = '/data/rj21/Data/Data_emidec'  # Linux bioeng358
    data_list = Util.CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    data_list_6_train = data_list
    
    # StT UNLABELLED
    path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
    data_list = Util.CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ))
    data_list_3_train = data_list
    random.shuffle(data_list_3_train)
    
    diceTe_Clin=[]
    
    # num_iter = 60
    batchTr = batch
    D1 = np.zeros((len(data_list_1_train),2))
    D1[:,0] = np.arange(0,len(data_list_1_train))
    D4 = np.zeros((len(data_list_4_train),2))
    D4[:,0] = np.arange(0,len(data_list_4_train))
    D5 = np.zeros((len(data_list_5_train),2))
    D5[:,0] = np.arange(0,len(data_list_5_train))
    D6 = np.zeros((len(data_list_6_train),2))
    D6[:,0] = np.arange(0,len(data_list_6_train))
    
    
    mu1, sigma1 = len(data_list_1_train)/10 ,  sigma*len(data_list_1_train)
    mu4, sigma4 = len(data_list_4_train)/10 ,  sigma*len(data_list_4_train)
    mu5, sigma5 = len(data_list_5_train)/10 ,  sigma*len(data_list_5_train)
    mu6, sigma6 = len(data_list_6_train)/10 ,  sigma*len(data_list_6_train)
    
    for epch in range(0,50):
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
        Inds1=[]
        Inds2=[]
        Inds4=[]
        Inds5=[]
        Inds6=[]
            
        for num_ite in range(0,10):
          
        ## Pro StT our dataset JOINT
            Indx = Util.rand_norm_distrb(batchTr, mu1, sigma1, [0,len(data_list_1_train)]).astype('int')
            Indx = D1[Indx,0].astype('int')
            sub_set = list(map(data_list_1_train.__getitem__, Indx))
            
            params = (128,  80,120,  -170,170,  -10,10,-10,10)
            loss_Joint, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
            dice = Util.dice_coef_batch( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
            diceTr1.append(dice.detach().cpu().numpy())
            Inds1.append(Indx)
            
        ### Pro StT our dataset CLINICAL
            Indx = Util.rand_norm_distrb(batchTr, mu4, sigma4, [0,len(data_list_4_train)]).astype('int')
            Indx = D4[Indx,0].astype('int')
            sub_set =list(map(data_list_4_train.__getitem__, Indx))
            
            loss_Clin, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
            dice = Util.dice_coef_batch( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
            diceTr4.append(dice.detach().cpu().numpy())
            Inds4.append(Indx)
            
        ### Pro MyoPS datatse
            Indx = Util.rand_norm_distrb(batchTr, mu5, sigma5, [0,len(data_list_5_train)]).astype('int')
            Indx = D5[Indx,0].astype('int')
            sub_set =list(map(data_list_5_train.__getitem__, Indx))
            
            loss_MyoPS, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
            dice = Util.dice_coef_batch( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
            diceTr5.append(dice.detach().cpu().numpy())
            Inds5.append(Indx)    
            
        ## Pro Emidec datatse
            Indx = Util.rand_norm_distrb(batchTr, mu6, sigma6, [0,len(data_list_6_train)]).astype('int')
            Indx = D6[Indx,0].astype('int')
            sub_set =list(map(data_list_6_train.__getitem__, Indx))
            
            loss_Emidec, res1, Imgs1, Masks1 = Network.Training.straightForward(sub_set, net, params, TrainMode=True, Contrast=False)
                                                       
            dice = Util.dice_coef_batch( res1[:,0,:,:]>0.5, Masks1[:,0,:,:].cuda() )                
            diceTr6.append(dice.detach().cpu().numpy())
            Inds6.append(Indx) 
        
        ## Consistency regularization
            Indx = np.random.randint(0,len(data_list_3_train),(batchTr,)).tolist()
            sub_set = list(map(data_list_3_train.__getitem__, Indx))
            loss_cons, Imgs_P, res, res_P = Network.Training.Consistency(sub_set, net, params, TrainMode=True, Contrast=False)
            diceTr3.append(1 - loss_cons.detach().cpu().numpy())
            
        ## backF - training
            net.train(mode=True)
            if epch>0:
                
                loss = 0.3*loss_Joint + loss_Clin + 0.3*loss_MyoPS + 0.1*loss_Emidec  + lamda_cons*loss_cons
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    
        if epch>0:
            scheduler.step()
            
        net.train(mode=False)
        
        ### StT lab
        params = (128,  80,120,  -0,0,  0,0,0,0)
        batchTE = 128
        random.shuffle(data_list_4_test)
        for num in range(0,4):   
            sub_set = data_list_4_test[num:num+batchTE]
            with torch.no_grad():
                _, resTE, ImgsTe, MasksTE = Network.Training.straightForward(sub_set, net, params, TrainMode=False, Contrast=False)       
                             
            dice = Util.dice_coef( resTE[:,0,:,:]>0.5, MasksTE[:,0,:,:].cuda() )                
            diceTe4.append(dice.detach().cpu().numpy())
             
            # for b in range(0,batch):
            #     A = resTE[b,0,:,:].detach().cpu().numpy()>0.5
            #     B = MasksTE[b,0,:,:].detach().cpu().numpy()>0.5
            #     HD4.append (np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A))))
        
        
        # print(np.mean(diceTe4))
        
                         
        D1 = D1[D1[:, 0].argsort()]
        D1[np.concatenate(np.array(Inds1)),1] = np.concatenate(np.tile(np.array(diceTr1),(batchTr,1)).T)
        D1 = D1[D1[:, 1].argsort()]
        D4 = D4[D4[:, 0].argsort()]
        D4[np.concatenate(np.array(Inds4)),1] = np.concatenate(np.tile(np.array(diceTr4),(batchTr,1)).T)
        D4 = D4[D4[:, 1].argsort()]
        D5 = D5[D5[:, 0].argsort()]
        D5[np.concatenate(np.array(Inds5)),1] = np.concatenate(np.tile(np.array(diceTr5),(batchTr,1)).T)
        D5 = D5[D5[:, 1].argsort()]
        D6 = D6[D6[:, 0].argsort()]
        D6[np.concatenate(np.array(Inds6)),1] = np.concatenate(np.tile(np.array(diceTr6),(batchTr,1)).T)
        D6 = D6[D6[:, 1].argsort()]
     
        torch.cuda.empty_cache()
    

  
    
    return np.mean(diceTe4)
        



# ---------------- Optimaliyace -----------------


# param_names=['lr']
# bounds_lw=[0.00001]
# bounds_up=[0.0001]
# pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up)))  

    
pbounds = {'lr':[0.00001,0.01],
           'batch':[28,128],
           'sigma':[0.7,1.2],
           'step_size':[15,40],
           'lambda_cons':[0.001,0.01]
           }  

optimizer = BayesianOptimization(f = get_value, pbounds=pbounds,random_state=1)  

optimizer.maximize(init_points=3,n_iter=40)

print(optimizer.max)

params=optimizer.max['params']
# print(params)

# 5:57

file_name = "BO_Unet_v7.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(optimizer, open_file)
# pickle.dump(params, open_file)
# open_file.close()

open_file = open(file_name, "wb")
pickle.dump(params, open_file)
open_file.close()

# open_file = open(file_name, "rb")
# data_list_test = pickle.load(open_file)
# open_file.close()