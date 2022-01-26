#for training Unet of ACDC - ED, ES
## data2 training 

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
import Loaders
import Unet_2D



# net = Unet_2D.UNet(enc_chs=(1,64,128,256,512), dec_chs=(512,256,128,64), out_sz=(128,128), retain_dim=False, num_class=2)
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_6.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00054, weight_decay=0.00001)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)


file_name = "data_list_Data2_all_dcm.pkl"
open_file = open(file_name, "rb")
data_list_test = pickle.load(open_file)
open_file.close()

b=2200
data_list_train = data_list_test[1:b]
data_list_test = data_list_test[b+1:-1]

# augmentation
params=[]

train_loss=[]
train_Dice=[]
test_Dice=[]
diceTe=[]
diceTr=[]

for epch in range(0,80):
    random.shuffle(data_list_train)
    net.train(mode=True)
    batch = 16
    diceTr=[]
    diceTe=[]
        
    for num in range(0,len(data_list_train)-batch-1, batch):

        
        t=0
        Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((batch,2,128,128) ), dtype=torch.float32)
        
        for b in range(0,batch):
            current_index = data_list_train[num+b]['slice']
            img_path = data_list_train[num+b]['img_path']
            mask_path = data_list_train[num+b]['mask_path']
    
            # img = Loaders.read_nii( img_path, (0,0,current_index,t) )
            # mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
            dataset = dcm.dcmread(img_path)
            img = dataset.pixel_array.astype(dtype='float32')
            dataset = dcm.dcmread(mask_path)
            mask = dataset.pixel_array
            mask = mask==1
            
            img = np.expand_dims(img, 0).astype(np.float32)
            mask = np.expand_dims(mask, 0).astype(np.float32)
        
            params=[]
            params.append({'Output_size': 128,
                           'Crop_size': random.randint(80,128),
                           'Angle': random.randint(-170,170),
                           'Transl': (random.randint(-10,10),random.randint(-10,10)),
                           'Scale': random.uniform(1.0,1.4)
                           })

            # plt.imshow(img[0,:,:],cmap='gray')
            # plt.show()
            
            img = Util.augmentation2(torch.tensor(img), params)
            mask = Util.augmentation2(torch.tensor(mask), params)
            mask = mask>0.25
            
            # plt.imshow(mask[0,:,:],cmap='gray')
            # plt.show()
            # plt.imshow(img[0,:,:],cmap='gray')
            # plt.show()


            # img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='Rand')
            # mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)
    
            # rot = random.randint(1,4)
            # img = np.rot90(img,rot,(0,1))
            # mask = np.rot90(mask,rot,(0,1))
            
            # if np.random.random()>0.5:
            #     img = np.fliplr(img)
            #     mask = np.fliplr(mask)
            
            # img = (img - np.mean(img))/ np.std(img)
        
    
            Imgs[b,0,:,:] = img
            Masks[b,0,:,:] = mask
        
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
        
        
        Masks[:,1,:,:] = (1-Masks[:,0,:,:])
        # Masks[:,0,:,:] = Masks[:,0,:,:]*2
        
        # weight = torch.tensor([0.88,0.18]).cuda()
        # loss = Util.dice_loss(res, Masks.cuda() )
        # loss = torch.nn.CrossEntropyLoss(weight)(res,  Masks.cuda() )
        # loss = -torch.mean( torch.log( torch.cat( (res[Masks==1], res[Masks==2]/20 ), 0 ) )  )
        # loss = -torch.mean( torch.log( res[Masks==1] ))
        loss = Util.dice_loss( res[:,0,:,:], Masks[:,0,:,:].cuda() )
        # loss = loss1 + loss2
                                                   
        train_loss.append(loss.detach().cpu().numpy())
    
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        optimizer.step()
                
        dice = Util.dice_coef( res[:,0,:,:]>0.25, Masks[:,0,:,:].cuda() )                
        diceTr.append(dice.detach().cpu().numpy())
        
        
        
        torch.cuda.empty_cache()


    scheduler.step()
    
    batch = 200
    net.train(mode=False)
    random.shuffle(data_list_test)

    for num in range(0,len(data_list_test)-batch-1, batch):
       
        t=0
        Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((batch,2,128,128) ), dtype=torch.float32)

        
        for b in range(0,batch):
            current_index = data_list_test[num+b]['slice']
            img_path = data_list_test[num+b]['img_path']
            mask_path = data_list_test[num+b]['mask_path']
        
            # img = Loaders.read_nii( img_path, (0,0,current_index,t) )
            # mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
            dataset = dcm.dcmread(img_path)
            img = dataset.pixel_array
            dataset = dcm.dcmread(mask_path)
            mask = dataset.pixel_array
            mask = mask==1
        
            img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='None')
            mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)
            
            img = torch.tensor(np.expand_dims(img, 0).astype(np.float32))
            mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32))
            
            # params=[]
            # params.append({'Output_size': 128,
            #                'Crop_size': random.randint(80,80),
            #                'Angle': random.randint(0,0),
            #                'Transl': (random.randint(0,0),random.randint(0,0)),
            #                'Scale': random.uniform(1.2,1.2)
            #                })
            
            # img = Util.augmentation2(img, params)
            # mask = Util.augmentation2(mask, params)
            # mask = mask>0.25
    
            Imgs[b,0,:,:] = img
            Masks[b,0,:,:] = mask
        
        
        with torch.no_grad(): 
            res = net( Imgs.cuda() )
            res = torch.softmax(res,dim=1)
                         
        dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTe.append(dice.detach().cpu().numpy())
        
        torch.cuda.empty_cache()
    
    train_Dice.append(np.mean(diceTr))
    test_Dice.append(np.mean(diceTe))
    
    print(np.mean(diceTe))
    
    plt.figure
    plt.imshow(Imgs[2,0,:,:].detach().numpy(), cmap='gray')
    plt.imshow(res[2,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    plt.show()
    
    plt.figure
    plt.plot(train_loss)
    plt.ylim([0.0, 0.9])
    plt.show()
    
    plt.figure
    plt.plot(train_Dice)
    plt.plot(test_Dice)
    plt.show()

