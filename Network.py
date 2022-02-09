## U-net
 # for version 2

import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import pydicom as dcm
import Utilities as Util
import random
# import matplotlib.pyplot as plt
# from PIL import Image 

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        # self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.BN    = nn.BatchNorm2d(in_ch)
    
    def forward(self, x):
        x = self.conv1(self.BN(x))
        res = x
        # x = self.conv3(self.conv2(x))
        x = self.conv2(x)
        return self.relu(x) + res
        # return self.relu(x)


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
        self.relu       = nn.ReLU()
    
    def forward(self, x):
        ftrs = []
        m, s = torch.mean(x,(2,3)), torch.std(x,(2,3))
        x = (x - m[:,:,None, None]) / s[:,:,None,None]
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class BottleNeck(nn.Module):
    def __init__(self, chs=(1024,1024) ):
        super().__init__()
        self.conv1x1_1 =  nn.Conv2d(chs[0], chs[1], 1, padding=0, padding_mode='replicate')     
        self.conv1x1_2 =  nn.Conv2d(chs[0], chs[1], 1, padding=0, padding_mode='replicate')     
        self.DP =  nn.Dropout(p=0.2)
    def forward(self, x):
        
        return self.conv1x1_2(self.DP(self.conv1x1_1(x)))
   


class Net(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.bottleneck  = BottleNeck((enc_chs[-1],enc_chs[-1]))
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        OutBN = self.bottleneck(enc_ftrs[::-1][0])
        out      = self.decoder(OutBN, enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
    
class Training(): 
    def straightForward(data_list, net, params, TrainMode=True): 

        net.train(mode=TrainMode)
        batch = len(data_list)
          
        Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((batch,2,128,128) ), dtype=torch.float32)
        
        for b in range(0,batch):
            current_index = data_list[b]['slice']
            img_path = data_list[b]['img_path']
            mask_path = data_list[b]['mask_path']
            t=0
            if img_path.find('.nii')>0:
                img = Util.read_nii( img_path, (0,0,current_index,t) )
                mask = Util.read_nii( mask_path, (0,0,current_index,t) )
                mask = mask==2
            elif img_path.find('.dcm')>0:
                dataset = dcm.dcmread(img_path)
                img = dataset.pixel_array.astype(dtype='float32')
                dataset = dcm.dcmread(mask_path)
                mask = dataset.pixel_array
                mask = mask==1  
        
            augm_params=[]
            augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                            'Scale': random.uniform(1.0,1.0),
                            'Flip': np.random.random()>0.5
                            })
                
            img = np.expand_dims(img, 0).astype(np.float32)
            mask = np.expand_dims(mask, 0).astype(np.float32)     
            
            img = Util.augmentation2(torch.tensor(img), augm_params)
            mask = Util.augmentation2(torch.tensor(mask), augm_params)
            mask = mask>0.5                      
    
            Imgs[b,0,:,:] = img
            Masks[b,0,:,:] = mask
        
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
        Masks[:,1,:,:] = (1-Masks[:,0,:,:])
        loss = Util.dice_loss( res[:,0,:,:], Masks[:,0,:,:].cuda() )
        
        return loss, res, Imgs, Masks
    


    def Consistency(data_list, net, params, TrainMode=True): 
        
        batch = len(data_list)
       
        Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
        # Imgs_P = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
    
        for b in range(0,batch):
            current_index = data_list[b]['slice']
            img_path = data_list[b]['img_path']
            t=0
            if img_path.find('.nii')>0:
                img = Util.read_nii( img_path, (0,0,current_index,t) ).astype(np.float32)   
            elif img_path.find('.dcm')>0:
                dataset = dcm.dcmread(img_path)
                img = dataset.pixel_array.astype(dtype='float32')
                
            img = Util.resize_with_padding(img,(128,128))      
            img =  np.expand_dims(img, 0).astype(dtype='float32')
            Imgs[b,0,:,:] = torch.tensor(img)
         
        augm_params=[]
        augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                            'Scale': random.uniform(1.0,1.0),
                            'Flip': np.random.random()>0.5
                            })
                   
        Imgs_P = Util.augmentation2(Imgs, augm_params)
        Imgs_P = torch.nan_to_num(Imgs_P, nan=0.0)
        
        net.train(mode=False)
    # with _torch.no_grad(): 
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
        res_P = net( Imgs_P.cuda() )
        res_P = torch.softmax(res_P,dim=1)
     
        res = Util.augmentation2(res[:,[0],:,:], augm_params)
        # MSE = nn.MSELoss()
        # loss = MSE(res, res_P[:,[0],:,:])
        loss = Util.dice_loss( res, res_P[:,[0],:,:] )
        # return loss, res, Imgs, Masks    
        return loss, Imgs_P, res, res_P
