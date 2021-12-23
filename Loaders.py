import numpy as np
import SimpleITK as sitk


def read_nii(file_name, current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)    
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
    extract_size = (sizeA[0], sizeA[1], 1, 1)
        
    # view = [1,2,0] # for sagital view
    # current_index = current_index[view[0], view[1], view[2]]
    # extract_size = extract_size[view[0], view[1], view[2]]
    # addX = [0,0,0,0]
    
    # for k in range(2):
    #     if extract_size[k] > sizeA[k]:
    #         addX[k*2] = int(np.floor(np.abs(sizeA[k]-extract_size[k])/2))
    #         addX[k*2+1] = int(np.ceil(np.abs(sizeA[k]-extract_size[k])/2))
    #         extract_size[k] = sizeA[k]
            
      
    # for k in range(3):
        # if current_index[k]==-1:
        #     current_index[k]=0
        #     extract_size[k]=size[k]
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    img = sitk.GetArrayFromImage(file_reader.Execute())
    # img = np.transpose(img, (view[0], view[1], view[2])) # for sagital view
    img = np.squeeze(img)
    

    # img = np.pad(img,((addX[2],addX[3]),(addX[0],addX[1]),(0,0)),'constant',constant_values=(-1024, -1024))


    return img