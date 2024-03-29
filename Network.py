## U-net
 # for version 2

import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import pydicom as dcm
import Utilities as Util
from torch.nn import init
import random
import matplotlib.pyplot as plt
import nibabel as nib
# from PIL import Image 



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(    
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )


    def forward(self,x):
        x = self.conv1(x)
        old_x = x
        x = self.conv2(x)
        return x + old_x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        x = self.conv3(x)
        return x
    
 

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        # self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024,ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        
        x[x==0] = float('nan')

        # ax = plt.hist(x[0,0,:,:].detach().cpu().numpy().ravel(), bins = 256)
        # plt.show()
        
        m, s = torch.nanmean(x,(2,3)), torch.tensor(np.nanstd(x.cpu().numpy(),(2,3))).cuda()
        x = (x - m[:,:,None, None]) / s[:,:,None,None]
        
        x[torch.isnan(x)] = 0
        
        # ax = plt.hist(x[0,0,:,:].detach().cpu().numpy().ravel(), bins = 256)
        # plt.xlim([-1.0, 1.0])
        # plt.show()

        
        
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)        
        # d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(x4)
        # d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        # plt.figure()
        # plt.imshow(np.mean(x1[0,:,:,:].detach().cpu().numpy(),0),vmin=0, vmax=1)
        # plt.show()
        
        return d1

    
class Training(): 
    def straightForward(data_list, net, params, TrainMode=True, Contrast=False): 

        net.train(mode=TrainMode)
        batch = len(data_list)
        vel = params[0]
        # vel = 256
          
        Imgs = torch.tensor(np.zeros((batch,1,vel,vel) ), dtype=torch.float32)
        Masks = torch.tensor(np.zeros((batch,1,vel,vel) ), dtype=torch.float32)
        
        for b in range(0,batch):
            current_index = data_list.iloc[b]['Slice']
            img_path = data_list.iloc[b]['Data_path']
            mask_path = img_path.replace('Data.','Mask.')
            t=0
            if img_path.find('.nii')>0:
                nii_data = nib.load(img_path)
                img = nii_data.dataobj[:,:,current_index]
                nii_dataM = nib.load(mask_path)
                mask = nii_dataM.dataobj[:,:,current_index]
                # mask = mask==1
            elif img_path.find('.dcm')>0:
                dataset = dcm.dcmread(img_path)
                img = dataset.pixel_array.astype(dtype='float32')
                dataset = dcm.dcmread(mask_path)
                mask = dataset.pixel_array
                mask = mask==1  
            
            resO = tuple(nii_data.header['pixdim'][1:4])
            # if len(dataset.dir('PixelSpacing'))>0:
            #     resO = (dataset['PixelSpacing'].value[0:2])
            # else:
            #     resO = (1.0, 1.0)
            
            resN = (params[11],params[11])
            
            ## Resamplinmg to 1mm 
            img = torch.tensor( np.expand_dims(img, 0).astype(np.float32))
            mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32) )             
            img = Util.Resampling(img, resO, resN, 'bilinear').detach().numpy()[0,:,:]
            mask = Util.Resampling(mask, resO,  resN, 'nearest').detach().numpy()[0,:,:]
            
            augm_params=[]
            augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                            'Scale': random.uniform(params[9],params[10]),
                            'Flip':  np.random.random()>0.5
                            })
            # print(augm_params[0]['Angle'])         
            
            augm = random.uniform(0, 1)>=0.3
            # augm = True
            
            if not augm:
                img = Util.resize_with_padding(img,(vel,vel))
                mask = Util.resize_with_padding(mask,(vel,vel))    
            
            img = np.expand_dims(img, 0).astype(np.float32)
            mask = np.expand_dims(mask, 0).astype(np.float32)    
            
            img = torch.tensor(img)
            mask = torch.tensor(mask)    

            
            if  augm:
                img = Util.augmentation2(img, augm_params)
                mask = Util.augmentation2(mask, augm_params)
                mask = mask>0.5   
                
        
            Imgs[b,0,:,:] = img
            Masks[b,0,:,:] = mask
        

        # if random.uniform(0, 1)>0.5:
        #     phi = random.uniform(0,2*np.pi)
        #     Imgs = Util.random_contrast(Imgs, [0.2, 3, phi])   
        
            
        res = net( Imgs.cuda() )
        # res = torch.softmax(res,dim=1)
        res = torch.sigmoid(res)
        loss = Util.dice_loss( res[:,0,:,:], Masks[:,0,:,:].cuda() )
        
        return loss, res, Imgs, Masks
    
  
    def Consistency(data_list, net, params, TrainMode=True, Contrast=False): 
        
        batch = len(data_list)
        vel = params[0]
        augm_params=[]
        augm_params.append({'Output_size': params[0],
                        'Crop_size': random.randint(params[1],params[2]),
                        'Angle': random.randint(params[3],params[4]),
                        'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                        'Scale': random.uniform(params[9],params[10]),
                        'Flip':  np.random.random()>0.5
                        })
       
        Imgs = torch.tensor(np.zeros((batch,1,vel,vel) ), dtype=torch.float32)
        Imgs_P = torch.tensor(np.zeros((batch,1,vel,vel) ), dtype=torch.float32)
        
        for b in range(0,batch):
            
            current_index = data_list.iloc[b]['Slice']
            img_path = data_list.iloc[b]['Data_path']
            mask_path = img_path.replace('Data.','Mask.')
            t=0
            if img_path.find('.nii')>0:
                nii_data = nib.load(img_path)
                img = nii_data.dataobj[:,:,current_index]
                nii_dataM = nib.load(mask_path)
                mask = nii_dataM.dataobj[:,:,current_index]
                # mask = mask==1
            elif img_path.find('.dcm')>0:
                dataset = dcm.dcmread(img_path)
                img = dataset.pixel_array.astype(dtype='float32')
                dataset = dcm.dcmread(mask_path)
                mask = dataset.pixel_array
                mask = mask==1  
            
            resO = tuple(nii_data.header['pixdim'][1:4])
            # if len(dataset.dir('PixelSpacing'))>0:
            #     resO = (dataset['PixelSpacing'].value[0:2])
            # else:
            #     resO = (1.0, 1.0)
            
            resN = (params[11],params[11])
            
            # resampling to 1mm resolution
            img = torch.tensor( np.expand_dims(img, 0).astype(np.float32))
            img = Util.Resampling(img, resO, resN, 'bilinear').detach().numpy()[0,:,:]
                
            # print(augm_params[0]['Angle'])

            Imgs[b,0,:,:] = torch.tensor( Util.resize_with_padding(img,(vel,vel)))

            img = torch.tensor(np.expand_dims(img, 0).astype(np.float32)  )
            
            # a=[]; a.append( augm_params[b])                  
            # img_P = Util.augmentation2(img, a)
            img_P = Util.augmentation2(img, augm_params)

            Imgs_P[b,0,:,:] = img_P
            
        
        net.train(mode=TrainMode)
        if TrainMode:
            res = net( Imgs.cuda() )
            res = torch.sigmoid(res)
            res_P = net( Imgs_P.cuda() )
            res_P = torch.sigmoid(res_P)
        else:           
            with torch.no_grad():
                res = net( Imgs.cuda() )
                res = torch.sigmoid(res)
                res_P = net( Imgs_P.cuda() )
                res_P = torch.sigmoid(res_P)
        
        res = Util.augmentation2(res[:,[0],:,:], augm_params)
        # # MSE = nn.MSELoss()
        # # loss = MSE(res, res_P[:,[0],:,:])
        loss = Util.dice_loss( res, res_P[:,[0],:,:] )
 
        return loss, Imgs_P, res, res_P
        # return loss, Imgs_P, res, res_P