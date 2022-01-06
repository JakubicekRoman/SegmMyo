#for training Unet of ACDC - ED, ES
import os
import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
import torch
# from torch.utils import data
# import torch.optim as optim
# import glob
# import random
# import torchvision.transforms as T
import pandas as pd
import pydicom as dcm
import cv2
# import skimage
import pickle

import Utilities as Util
import Loaders
import Unet_2D


## validation on testing data

# net = Unet_2D.UNet(enc_chs=(1,64,128,256,512), dec_chs=(512,256,128,64), out_sz=(128,128), retain_dim=False, num_class=2)
# net = torch.load(r"D:\jakubicek\SegmMyo\Models\net_v1_3.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_3.pt")
net = net.cuda()


## -------------- validation for OUR data ---------------
path_data = '/data/rj21/MyoSeg/Data'  # Linux bioeng358
# path_data = 'D:\jakubicek\SegmMyo\Clinical_Database_StThomas'  # Win CUDA2
data_list_test = Util.CreateDatasetOur(os.path.normpath( path_data ))

file_name = "data_list_OUR_T1_pre.pkl"
open_file = open(file_name, "wb")
pickle.dump(data_list_test, open_file)
open_file.close()
# open_file = open(file_name, "rb")
# data_list_test = pickle.load(open_file)
# open_file.close()


res_table = pd.DataFrame(data=[], columns=['FileName', 'Slice' ,'Info', 'SizeData' ,'ID_image','FilePath'] )                                         

test_Dice=[]
diceTe=[]
    
batch = 1
net.train(mode=False)

for num in range(0,len(data_list_test)):
# for num in range(0,300):    
   
    t=0
    Imgs = torch.tensor(np.zeros((batch,1,128,128) ), dtype=torch.float32)
    Masks = torch.tensor(np.zeros((batch,2,128,128) ), dtype=torch.float32)

    
    for b in range(0,batch):
        current_index = data_list_test[num+b]['slice']
        img_path = data_list_test[num+b]['img_path']
        mask_path = data_list_test[num+b]['mask_path']
        nPat = data_list_test[num+b]['Patient']
    
        dataset = dcm.dcmread(img_path)
        info = dataset.SeriesDescription
        img = dataset.pixel_array
    
        img, transl = Util.augmentation(img, new_width=128, new_height=128, rand_tr='None')
        # mask, _  = Util.augmentation(mask, new_width=128, new_height=128, rand_tr = transl)
        
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0).astype(np.float32)
        # mask = np.expand_dims(mask, 0).astype(np.float32)

        Imgs[b,0,:,:] = torch.tensor(img)
        # Masks[b,0,:,:] = torch.tensor(mask)
    
    
    with torch.no_grad(): 
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)
                     
    # dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )                
    # diceTe.append(dice.detach().cpu().numpy())
    
    torch.cuda.empty_cache()

    # test_Dice.append(np.mean(diceTe))
    
    ID = '0000000' + str(num)
    ID = ID[-3:]
    
    res_table.loc[(num,'FilePath')] =   img_path
    res_table.loc[(num,'FileName')] =   nPat
    res_table.loc[(num,'ID_image')] =  ID
    res_table.loc[(num,'Slice')] = current_index
    res_table.loc[(num,'Info')] = info
    res_table.loc[(num,'SizeData')] = data_list_test[num+b]['Size']
    # res_table.loc[(num,'Dice')] = dice.detach().cpu().numpy()
    
    # path_save = 'valid\\Main_1\StThomas\T1_pre'
    path_save = '/data/rj21/MyoSeg/valid/Main_1/T1_post'
    # path_save = '/data/rj21/MyoSeg/valid/Main_1/T2'

            

    resB = (res[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    ctr = dimg - resB
    
    imgB = Imgs[0,0,:,:].detach().numpy()
    # m, s = np.mean(imgB,(0,1)), np.std(imgB,(0,1))
    # imgB = (imgB - m) / s
    imgB = ( imgB - imgB.min() ) / (imgB.max() - imgB.min())
    imgB = ( imgB - 0.0 ) / (0.5 - 0.0)
    imgB[imgB>1.0]=1.0
    RGB_imgB = cv2.cvtColor(imgB,cv2.COLOR_GRAY2RGB)
    comp = RGB_imgB[:,:,1]
    comp[ctr==1]=1
    RGB_imgB[:,:,1] = comp
    
    
    Fig = plt.figure()
    plt.imshow(RGB_imgB)
    
    plt.show()
    plt.draw()
    
    # Fig.savefig( path_save + '/' + 'res_' + nPat +  '_'  + ID + '.png')
    # plt.close(Fig)

    # Util.save_to_excel(res_table, path_save + '/' , 'ResultsDet')

