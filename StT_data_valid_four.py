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
# import Network_v9 as Network
import Network as Network

data_list_test=[]

# ## StT LABELLED - JOINT
path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
data_list = Loaders.CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
data_list_test = data_list_test + data_list
# data_list_test = data_list_test + data_list[1028:1154]

# # ## StT LABELLED - P1-30
# path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
# data_list_test = data_list_test + data_list
# # b = int(len(data_list)*0.55)
# # data_list_test = data_list[b+1:]

## StT LABELLED - Alina data T2
# # path_data = '/data/rj21/Data/Data_T2_Alina/dcm_resaved'  # Linux bioeng358
# path_data = '/data/rj21/Data/Data_1mm/T2_alina'
# data_list = Loaders.CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
# # data_list_test = data_list
# data_list_test = data_list_test + data_list

# b = int(len(data_list)*0.55)
# data_list_test = data_list[b+1:]

# b = int(len(data_list)*0.7)
# data_list_train = data_list[1:b]
# data_list_test = data_list[b+1:-1]

# file_name = "data_list_Data2_all_dcm.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(data_list_test, open_file)
# open_file.close()
# open_file = open(file_name, "rb")
# data_list_test = pickle.load(open_file)
# open_file.close()

# version = "v3_1_9_5"
# version = "v3_3_4"
version = "v8_4_2"
# version = "v9_2_3"

# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v3_0_0.pt")
# net = torch.load(r"/data/rj21/MyoSeg/Models/net_v1_5.pt")
net = torch.load(r"/data/rj21/MyoSeg/Models/net_" + version + ".pt")
net = net.cuda()

# path_save = '/data/rj21/MyoSeg/valid/Main_8_1mm'
path_save = '/data/rj21/MyoSeg/valid/Main_v9'

# save_name = 'Joint_valid'
# save_name = 'All_valid'
save_name = 'joint_valid'


# path_save = '/data/rj21/MyoSeg/valid'
try:
    os.mkdir(path_save)
except OSError as error:
    print(error) 

res_table = pd.DataFrame(data=[], columns=['Name','ID_pat','Slice' ,'Dataset','Seq', 'Dice','HD', 'ID_image'] )                                         

diceTe=[]
vel=[]
iii=0
velImg = 256;


for num in range(0,len(data_list_test),1):
# for num in range(0,100,1):    

    Imgs = torch.tensor(np.zeros((1,4,velImg,velImg) ), dtype=torch.float32)
    Masks = torch.tensor(np.zeros((1,4,velImg,velImg) ), dtype=torch.float32)

    
    current_index = data_list_test[num]['slice']
    img_path1 = data_list_test[num]['img_path']
    mask_path1 = data_list_test[num]['mask_path']
    nPat = data_list_test[num]['pat_name']
    file_name = data_list_test[num]['file_name']
    seq = data_list_test[num]['Seq']

    # img = Loaders.read_nii( img_path, (0,0,current_index,t) )
    # mask = Loaders.read_nii( mask_path, (0,0,current_index,t) )
    
    nImg = ('T1','T2','W1','W4')
    for c in range(0,4):
    # for c in range(0,1):
        
        img_path = img_path1.replace('W4',nImg[c])
        mask_path = mask_path1.replace('W4',nImg[c])
        
        dataset = dcm.dcmread(img_path)
        img = dataset.pixel_array
        dataset = dcm.dcmread(mask_path)
        mask = dataset.pixel_array
        mask = mask==1
        
        vel.append(img.shape)
    
        # img, p_cut, p_pad = Util.crop_center_final(img,velImg,velImg)
        # mask, p_cut, p_pad = Util.crop_center_final(mask,velImg,velImg)
    
        img = torch.tensor(np.expand_dims(img, 0).astype(np.float32))
        mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32))
           
        params = (256,  255,256,  0,0,  0,0,0,0)
        augm_params=[]
        augm_params.append({'Output_size': params[0],
                            'Crop_size': random.randint(params[1],params[2]),
                            'Angle': random.randint(params[3],params[4]),
                            'Transl': (random.randint(params[5],params[6]),random.randint(params[7],params[8])),
                            'Scale': random.uniform(1.0,1.0),
                            'Flip': False
                            })
        img = Util.augmentation2(img, augm_params)
        mask = Util.augmentation2(mask, augm_params)
        mask = mask>0.5
        
        Imgs[0,c,:,:] = img
        Masks[0,c,:,:] = mask
        
    
    with torch.no_grad(): 
        res = net( Imgs.cuda() )
        # res = torch.softmax(res,dim=1)
        res = torch.sigmoid(res)      # for V9_                
    
    torch.cuda.empty_cache()
        
    dice = Util.dice_coef( res[:,0,:,:]>0.5, Masks[:,0,:,:].cuda() )  
              
    A = res[0,0,:,:].detach().cpu().numpy()>0.5
    B = Masks[0,0,:,:].detach().cpu().numpy()>0.5
    HD = np.max((Util.MASD_compute(A,B),Util.MASD_compute(B,A)))

    
    res_table.loc[(num,'Name')] =   nPat 
    res_table.loc[(num,'ID_pat')] =   data_list_test[num]['ID_pat'] 
    res_table.loc[(num,'ID_scan')] = data_list_test[num]['ID_scan']
    res_table.loc[(num,'Slice')] =   current_index 
    # res_table.loc[(num,'Dataset')] =   file_name.split('_')[0]
    res_table.loc[(num,'Dataset')] =   file_name.split('_')[0] + 'A'
    res_table.loc[(num,'Seq')] =   seq
    res_table.loc[(num,'HD')] =   HD
    res_table.loc[(num,'Dice')] =   dice.item()
    res_table.loc[(num,'ID_image')] =   iii

    
    print(nPat)

    resB = (res[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    dimg = cv2.dilate(resB, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    ctr = dimg - resB
    
    GT = (Masks[0,0,:,:].detach().cpu().numpy()>0.5).astype(np.dtype('uint8'))
    dimg = cv2.dilate(GT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) )
    ctrGT = dimg - GT
    
    imgB = Imgs[0,0,:,:].detach().numpy()
    imgB = ( imgB - imgB.min() ) / (imgB.max() - imgB.min())
    RGB_imgB = cv2.cvtColor(imgB,cv2.COLOR_GRAY2RGB)
    
    comp = RGB_imgB[:,:,0]
    comp[ctr==1]=1
    comp[ctrGT==1]=0
    RGB_imgB[:,:,0] = comp
    
    comp = RGB_imgB[:,:,1]
    comp[ctr==1]=0
    comp[ctrGT==1]=1
    RGB_imgB[:,:,1] = comp
    
    comp = RGB_imgB[:,:,2]
    comp[ctr==1]=0
    comp[ctrGT==1]=0
    RGB_imgB[:,:,2] = comp
    
    # Fig = plt.figure()
    # plt.imshow(RGB_imgB)
    # plt.show()
    # plt.draw()
    
    # Fig.savefig( path_save + '/' + 'res_' + "%.4f" % (dice.item()) + '_' +  nPat + file_name.split('_')[0] + '_' + seq + '_' + current_index + '.png', dpi=150)
    # plt.close(Fig)
    

    
Util.save_to_excel(res_table, path_save + '/' , 'Res_' + save_name + '_' + version)