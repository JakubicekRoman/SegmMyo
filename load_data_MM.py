import pydicom as dcm
import matplotlib.pyplot as plt
import glob
import os

import Loaders


path_data = '/data/rj21/test/Data_MaM'

data_list = []
mask_list = []

data_list.append('/data/rj21/test/Data_MaM/A0S9V9/A0S9V9_sa.nii.gz')
mask_list.append('/data/rj21/test/Data_MaM/A0S9V9/A0S9V9_sa_gt.nii.gz')

# for dir_name in os.listdir(path_data):
#     if os.path.isdir(os.path.join(path_data, dir_name)):
#         sdir_list = os.listdir(os.path.join(path_data, dir_name))
#         sdir_list.sort()
#         # ssdir_name = ssdir_list[0]
        
#         for _, sdir_name in enumerate(sdir_list):
        
#             path1 = os.path.join(path_data, dir_name, sdir_name)
            
#             for slice_name in os.listdir(path1):
#             # for i in range(0,1):  
#             #     slice_name =  os.listdir(path1)[i]
                
#                 dataset = dcm.dcmread(os.path.join(path1, slice_name))
#                 t = dataset.SeriesDescription
#                 if t.find('T1')>=0:
#                     if t.find('PRE')>=0 or t.find('Pre')>=0:
#                         data_list.append(os.path.join(path1, slice_name))
    
#                         # print(os.path.join(path1, slice_name))
#                         # print(t)
#                 # data_list.append(os.path.join(path1, slice_name))

ind_slice = 3
time = 0

file_name = data_list[0]
img = Loaders.read_nii(file_name, (0,0,ind_slice,time) )

file_name = mask_list[0]
mask = Loaders.read_nii(file_name, (0,0,ind_slice,time) )

mask[mask>2]=0
mask[mask<2]=0

plt.Figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

plt.Figure()
plt.imshow(mask, cmap=plt.cm.gray)
plt.show()
            
# data_list.sort()

# for i in range(0,200):
#     dataset = dcm.dcmread(data_list[i])
    
#     # print(dataset.ProtocolName)
#     print(dataset.SeriesDescription)
    
#     # img = dataset.pixel_array

#     # plt.Figure()
#     # plt.imshow(img, cmap=plt.cm.gray)
#     # plt.show()
    
#     # print(i)
            

