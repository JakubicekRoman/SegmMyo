import pydicom as dcm
import matplotlib.pyplot as plt
import glob
import os

path_data = '/data/rj21/MyoSeg/Data'

# data_list = glob.glob( path_data + "/**/*.dcm", recursive=True)  

data_list = []

for dir_name in os.listdir(path_data):
    if os.path.isdir(os.path.join(path_data, dir_name)):
        sdir_list = os.listdir(os.path.join(path_data, dir_name))
        sdir_list.sort()
        # ssdir_name = ssdir_list[0]
        
        for _, sdir_name in enumerate(sdir_list):
        
            path1 = os.path.join(path_data, dir_name, sdir_name)
            
            for slice_name in os.listdir(path1):
            # for i in range(0,1):  
            #     slice_name =  os.listdir(path1)[i]
                
                dataset = dcm.dcmread(os.path.join(path1, slice_name))
                t = dataset.SeriesDescription
                
                # if t.find('T1')>=0:
                #     # data_list.append(os.path.join(path1, slice_name))
                    
                #     if t.find('post')>=0 or t.find('Post')>=0:
                #         data_list.append(os.path.join(path1, slice_name))
                        
                #     # if t.find('PRE')>=0 or t.find('Pre')>=0:
                #         # data_list.append(os.path.join(path1, slice_name))
               
                
                if t.find('T2')>=0:
                    data_list.append(os.path.join(path1, slice_name))
                        # print(os.path.join(path1, slice_name))
                        # print(t)
                
                # data_list.append(os.path.join(path1, slice_name))

data_list.sort()


# dataset = dcm.dcmread(data_list[25])
# img = dataset.pixel_array

# plt.Figure()
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()


for i in range(0,1000,10):        
    dataset = dcm.dcmread(data_list[i])
    
    print(dataset.ProtocolName)
    # print(dataset.SeriesDescription)
    
    img = dataset.pixel_array

    plt.Figure()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
    print(data_list[i])
            

