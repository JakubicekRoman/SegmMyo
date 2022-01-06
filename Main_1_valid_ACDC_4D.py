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
import pandas as pd
import cv2

import Utilities as Util
import Loaders
import Unet_2D



# net = Unet_2D.UNet(enc_chs=(1,64,128,256,512), dec_chs=(512,256,128,64), out_sz=(128,128), retain_dim=False, num_class=2)
# net = torch.load(r"D:\jakubicek\SegmMyo\Models\net_v1_3.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_3.pt")


net = net.cuda()

## -------------- validation for \ADCD ------------------
path_data = '/data/rj21/MyoSeg/Data_ACDC/training'  # Linux bioeng358
# path_data = 'D:\jakubicek\SegmMyo\Data_ACDC\\training'  # Win CUDA2
# data_list_train, data_list_test = Util.CreateDataset(os.path.normpath( path_data ))
data_list_test = Util.CreateDataset_4D(os.path.normpath( path_data ))

res_table = pd.DataFrame(data=[], columns=['FileName', 'Slice' ,'ID_image', 'Dice'] )                                         


test_Dice=[]
diceTe=[]
diceTr=[]
    
batch = 1
net.train(mode=False)
# random.shuffle(data_list_test)


for num in range(0,len(data_list_test)-batch-1, batch):
# for num in range(0,len(data_list_test)-batch-1, 2):
# for num in range(0,300):    
   
    t=0
    Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
    Masks = torch.tensor(np.zeros((batch,2,128,128) ), dtype=torch.float32)

    
    for b in range(0,batch):
        current_index = data_list_test[num+b]['slice']
        img_path = data_list_test[num+b]['img_path']
        # mask_path = data_list_test[num+b]['mask_path']
        t = data_list_test[num+b]['time']
        nPat = data_list_test[num+b]['Patient']
    
        img = Loaders.read_nii( img_path, (0,0,current_index,t) )
        # mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
        # mask = mask==2
    
        img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='None')
        # mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)

        img = np.expand_dims(img, 0).astype(np.float32)
        # mask = np.expand_dims(mask, 0).astype(np.float32)

        Imgs[b,0,:,:] = torch.tensor(img)
        # Masks[b,0,:,:] = torch.tensor(mask)
    
    
    with torch.no_grad(): 
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
                     
    dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
    diceTe.append(dice.detach().cpu().numpy())
    
    torch.cuda.empty_cache()

    test_Dice.append(np.mean(diceTe))
    
    ID = '0000000' + str(num)
    ID = ID[-6:]
    
    res_table.loc[(num,'FileName')] =   img_path 
    res_table.loc[(num,'ID_image')] =  ID
    res_table.loc[(num,'Slice')] = current_index
    res_table.loc[(num,'Dice')] = dice.detach().cpu().numpy()
    
    path_save = '/data/rj21/MyoSeg/valid/Main_1/ACDC'
    
    # Util.save_to_excel(res_table, path_save + '\\' , 'ResultsDet')
    
    resB = (res[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    ctr = dimg - resB
    
    imgB = Imgs[0,0,:,:].detach().numpy()
    imgB = ( imgB - imgB.min() ) / (imgB.max() - imgB.min())
    RGB_imgB = cv2.cvtColor(imgB,cv2.COLOR_GRAY2RGB)
    comp = RGB_imgB[:,:,1]
    comp[ctr==1]=1
    RGB_imgB[:,:,1] = comp
    
    
    Fig = plt.figure()
    plt.imshow(RGB_imgB)
    
    plt.show()
    plt.draw()
    
    ID = '0000000' + str(current_index)
    ID = ID[-3:]
    
    T = '0000000' + str(t)
    T = T[-3:]

    
    Fig.savefig( path_save + '/' + 'res_' + nPat + '_' + ID + '_' + T + '.png')
    plt.close()
            
     # Fig = plt.figure()
    # plt.imshow(Imgs[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(res[0,1,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    # plt.draw()
    # Fig.savefig( path_save + '\\' + 'Res_' + ID + '.png')
    # plt.close()
    
    # Fig = plt.figure()
    # plt.imshow(Imgs[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(res[0,1,:,:].detach().cpu().numpy()>0.5, cmap='jet', alpha=0.2)
    # plt.show()
    # plt.draw()
    # Fig.savefig( path_save + '\\' + 'Seg_' + ID + '.png')
    # plt.close()
    
    # Fig = plt.figure()
    # plt.imshow(Imgs[0,0,:,:].detach().numpy(), cmap='gray')
    # plt.imshow(1-Masks[0,0,:,:].detach().cpu().numpy(), cmap='jet', alpha=0.2)
    # plt.show()
    # plt.draw()
    # Fig.savefig( path_save + '\\' + 'GT_' + ID + '.png')
    # plt.close()


