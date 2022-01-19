#for training Unet of ACDC - ED, ES

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

import Utilities as Util
import Loaders
import Unet_2D



net = Unet_2D.UNet(enc_chs=(1,64,128,256,512), dec_chs=(512,256,128,64), out_sz=(128,128), retain_dim=False, num_class=2)
# net = torch.load(r"D:\jakubicek\SegmMyo\Models\net_v1_0.pt")

net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.000237, weight_decay=0.0000099)
# optimizer = optim.SGD(net2.parameters(), lr=0.000001, weight_decay=0.0001, momentum= 0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)



path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
# path_data = 'D:\jakubicek\SegmMyo\Data_ACDC\\training'  # Win CUDA2
data_list_train, data_list_test = Util.CreateDataset(os.path.normpath( path_data ))

train_loss=[]
train_Dice=[]
test_Dice=[]
diceTe=[]
diceTr=[]

for epch in range(0,50):
    random.shuffle(data_list_train)
    net.train(mode=True)
    batch = 13
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
        
        
            img = Loaders.read_nii( img_path, (0,0,current_index,t) )
            mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
            mask = mask==2
        
            img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='Rand')
            mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)
    
            rot = random.randint(1,4)
            img = np.rot90(img,rot,(0,1))
            mask = np.rot90(mask,rot,(0,1))
            
            if np.random.random()>0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            
            # img = (img - np.mean(img))/ np.std(img)
            
            img = np.expand_dims(img, 0).astype(np.float32)
            mask = np.expand_dims(mask, 0).astype(np.float32)
    
            Imgs[b,0,:,:] = torch.tensor(img)
            Masks[b,0,:,:] = torch.tensor(mask)
        
        # rotater = T.RandomRotation(degrees=(-60, 60))
        # rotated_imgs = rotater(Imgs)
        
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
        
        
        Masks[:,1,:,:] = (1-Masks[:,0,:,:])
        # Masks[:,0,:,:] = Masks[:,0,:,:]*2
        
        # loss = Util.dice_loss(res, Masks.cuda() )
        # loss = torch.nn.CrossEntropyLoss()(res[:,1,:,:],  Masks.type(torch.long).cuda() )
        # loss = -torch.mean( torch.log( torch.cat( (res[Masks==1], res[Masks==2]/20 ), 0 ) )  )
        # loss1 = -torch.mean( torch.log( res[Masks==1] ))
        loss = Util.dice_loss( res[:,0,:,:], Masks[:,0,:,:].cuda() )
        # loss = loss1 + loss2
                                                   
        train_loss.append(loss.detach().cpu().numpy())
    
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)
        optimizer.step()
                
        dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTr.append(dice.detach().cpu().numpy())
        
        torch.cuda.empty_cache()


    scheduler.step()
    
    batch = 102
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
        
            img = Loaders.read_nii( img_path, (0,0,current_index,t) )
            mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
            mask = mask==2
        
            img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='Rand')
            mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)
    
            img = np.expand_dims(img, 0).astype(np.float32)
            mask = np.expand_dims(mask, 0).astype(np.float32)
    
            Imgs[b,0,:,:] = torch.tensor(img)
            Masks[b,0,:,:] = torch.tensor(mask)
        
        
        with torch.no_grad(): 
            res = net( Imgs.cuda() )
            res = torch.softmax(res,dim=1)
                         
        dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
        diceTe.append(dice.detach().cpu().numpy())
        
        torch.cuda.empty_cache()
    
    train_Dice.append(np.mean(diceTr))
    test_Dice.append(np.mean(diceTe))
    
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

