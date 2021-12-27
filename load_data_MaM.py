import pydicom as dcm
import matplotlib.pyplot as plt
import glob
import os

import Loaders


path_data = '/data/rj21/MyoSeg/Data_MaM/Training/Labeled'

data_list = []
mask_list = []

# data_list.append('/data/rj21/MyoSeg/Data_MaM/A0S9V9/A0S9V9_sa.nii.gz')
# mask_list.append('/data/rj21/MyoSeg/Data_MaM/A0S9V9/A0S9V9_sa_gt.nii.gz')


for dir_name in os.listdir(path_data):
    if os.path.isdir(os.path.join(path_data, dir_name)):
        
        pat_path = os.path.join(path_data, dir_name)
        # print(pat_path)
        file_name = os.listdir(pat_path)
        for _,file in enumerate(file_name):
            # if file.find('gt.nii')>=0:
            if file.find('gt.nii')==-1:
                data_list.append(os.path.join( pat_path, file ))
            else:
                mask_list.append(os.path.join( pat_path, file ))
        
        # sizeData = Loaders.size_nii( os.path.join( pat_path, file_name[1] ))        
        # print(sizeData)
        

for pat in range(30,31):
    img_name = data_list[pat]
    mask_name = mask_list[pat]
                           
    sizeData = Loaders.size_nii( img_name )

    print(sizeData)
    
    for t in range(0,sizeData[3]):
        # for ind_slice in range(0,sizeData[2]):
        for ind_slice in range(5,6):
    # for ind_slice in range(0,sizeData[2]):
    #     # for t in range(0,sizeData[3]):
    #     for t in range(0,1):
            img = Loaders.read_nii(img_name, (0,0,ind_slice,t) )
            mask = Loaders.read_nii(mask_name, (0,0,ind_slice,t) )
    
            plt.Figure()
            plt.imshow(img, cmap=plt.cm.gray)
            # plt.imshow(mask, cmap='jet', alpha=0.2)
            plt.show()
    
# ind_slice = 3
# time = 0

# file_name = data_list[0]
# img = Loaders.read_nii(file_name, (0,0,ind_slice,time) )

# file_name = mask_list[0]
# mask = Loaders.read_nii(file_name, (0,0,ind_slice,time) )

# mask[mask>2]=0
# mask[mask<2]=0

# plt.Figure()
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# plt.Figure()
# plt.imshow(mask, cmap=plt.cm.gray)
# plt.show()
            
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
            

