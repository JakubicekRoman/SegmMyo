## dataloader for new data2 - annotated

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
import pickle
import pydicom as dcm

import Utilities as Util
import Loaders
import Unet_2D


def CreateDataset7(path_data):
    data_list_tr = []
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                # if file.find('Ref')>=0:
                    path_mask = os.path.join(path_data, pat_name, file)
                    name = file[0:file.find('_gt')] + file[file.find('_gt')+3:]
                    # path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                    path_maps = os.path.join(path_data, pat_name, name)

                    # sizeData = Loaders.size( path_maps )
                    sizeData = Loaders.size_nii( path_maps )

                    if len(sizeData)==2:
                        sizeData = sizeData + (1,)
                    # print(sizeData)
                    
                    for sl in range(0,sizeData[2]):
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'slice': name[-7:-4],
                                              'ID_pat': ii
                                              } )
            
    return data_list_tr


## -------------- validation for \StT data annotated ------------------
path_data = '/data/rj21/Data/Data2/Resaved_data_StT'  # Linux bioeng358
# path_data = 'D:\jakubicek\SegmMyo\Data_ACDC\\training'  # Win CUDA2
data_list_test = CreateDataset7(os.path.normpath( path_data ))

# file_name = "data_list_Data2_all_dcm.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(data_list_test, open_file)
# open_file.close()
# open_file = open(file_name, "rb")
# data_list_test = pickle.load(open_file)
# open_file.close()


# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_5.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_v2_2.pt")
net = net.cuda()

path_save = '/data/rj21/MyoSeg/valid'

res_table = pd.DataFrame(data=[], columns=['Name','ID_pat','Slice' ,'Dataset','Seq', 'Dice','HD'] )                                         

diceTe=[];

for num in range(0,len(data_list_test),1):
# for num in range(22,23,1):    

    Imgs = torch.tensor(np.zeros((1,1,128,128) ), dtype=torch.float32)
    Masks = torch.tensor(np.zeros((1,2,128,128) ), dtype=torch.float32)

    
    current_index = data_list_test[num]['slice']
    img_path = data_list_test[num]['img_path']
    mask_path = data_list_test[num]['mask_path']
    nPat = data_list_test[num]['pat_name']
    file_name = data_list_test[num]['file_name']

    # img = Loaders.read_nii( img_path, (0,0,current_index,t) )
    # mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
    dataset = dcm.dcmread(img_path)
    img = dataset.pixel_array
    dataset = dcm.dcmread(mask_path)
    mask = dataset.pixel_array
    mask = mask==1
    
    img = Util.crop_center(img, new_width=128, new_height=128)
    mask = Util.crop_center(mask, new_width=128, new_height=128)
    
    # print(np.shape(img))
    
    # Fig = plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.imshow(mask*0.2, cmap='cool', alpha=0.2)
    # plt.show()
    
    img = np.expand_dims(img, 0).astype(np.float32)
    Imgs[0,0,:,:] = torch.tensor(img)
    
    mask = np.expand_dims(mask, 0).astype(np.float32)
    Masks[0,0,:,:] = torch.tensor(mask)
    
    
    with torch.no_grad(): 
        res = net( Imgs.cuda() )
        res = torch.softmax(res,dim=1)                    
    
    torch.cuda.empty_cache()
        
    dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )  
              
    A = res[0,0,:,:].detach().cpu().numpy()>0.5
    B = Masks[0,0,:,:].detach().cpu().numpy()>0.5
    HD = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))

    
    res_table.loc[(num,'Name')] =   nPat 
    res_table.loc[(num,'ID_pat')] =   data_list_test[num]['ID_pat'] 
    res_table.loc[(num,'Slice')] =   current_index 
    res_table.loc[(num,'Dataset')] =   file_name.split('_')[0] 
    res_table.loc[(num,'Seq')] =   file_name.split('_')[1] 
    res_table.loc[(num,'HD')] =   HD     
    res_table.loc[(num,'Dice')] =   dice.detach().cpu().numpy()
    
    # print(data_list_test[num]['file_name'])

Util.save_to_excel(res_table, path_save + '/' , 'ResultsDet')


    # resB = (res[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    # dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    # ctr = dimg - resB
    
    # GT = (Masks[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    # dimg = cv2.dilate(GT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    # ctrGT = dimg - GT
    
    # imgB = Imgs[0,0,:,:].detach().numpy()
    # imgB = ( imgB - imgB.min() ) / (imgB.max() - imgB.min())
    # RGB_imgB = cv2.cvtColor(imgB,cv2.COLOR_GRAY2RGB)
    
    # comp = RGB_imgB[:,:,0]
    # comp[ctr==1]=1
    # # comp[ctrGT==1]=0
    # RGB_imgB[:,:,0] = comp
    
    # comp = RGB_imgB[:,:,1]
    # comp[ctr==1]=0
    # # comp[ctrGT==1]=1
    # RGB_imgB[:,:,1] = comp
    
    # comp = RGB_imgB[:,:,2]
    # comp[ctr==1]=0
    # # comp[ctrGT==1]=0
    # RGB_imgB[:,:,2] = comp
    
    # Fig = plt.figure()
    # plt.imshow(RGB_imgB)
    
    # plt.show()
    # plt.draw()
    
    # # Fig.savefig( path_save + '/' + 'res_' + nPat +  '_'  + str(current_index) + '.png')
    # plt.close(Fig)
    
    # Fig = plt.figure()
    # plt.imshow(imgB, cmap='gray')
    
    # plt.show()
    # plt.draw()
    
    # # Fig.savefig( path_save + '/' + 'res_' + nPat +  '_'  + str(current_index) + '_orig.png')
    # plt.close(Fig)
    
