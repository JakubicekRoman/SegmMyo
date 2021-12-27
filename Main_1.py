import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import glob

import Utilities as Util
import Loaders
import Unet_2D


net = Unet_2D.UNet(enc_chs=(1,64,128,256), dec_chs=(256,128,64), out_sz=(128,128), retain_dim=False, num_class=2)
net = net.cuda()


# path_data = '/data/rj21/MyoSeg/Data_ACDC/training'  # Linux bioeng358
path_data = 'D:\jakubicek\SegmMyo\Data_ACDC\\training'  # Win CUDA2

data_list_train, data_list_test = Util.CreateDataset(os.path.normpath( path_data ))

num = 4

t=0
current_index = data_list_train[num]['slice']
img_path = data_list_train[num]['img_path']

img = Loaders.read_nii( img_path, (0,0,current_index,t) )

img = Util.center_crop(img, new_width=128, new_height=128)

plt.figure
plt.imshow(img, cmap='gray')
plt.show()

img = np.expand_dims(img,0)
img = np.expand_dims(img,0).astype(np.float32)

img = torch.tensor(img).cuda()

res = net( img )

res = torch.softmax(res,dim=1)

res_img = res.detach().cpu().numpy()[0,1,:,:]

plt.figure
plt.imshow(res_img, cmap='jet')
plt.show()


