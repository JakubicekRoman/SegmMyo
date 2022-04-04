import numpy as np
import os
import pydicom as dcm

import matplotlib.pyplot as plt
from Utilities import size_nii




def CreateDataset_ACDC(path_data, ):      

    data_list_tr = []
    data_list_te = []
    pat = 0

    for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            
            pat_name = os.path.join(path_data, dir_name)
            file_name = os.listdir(pat_name)
            
            sizeData = size_nii( os.path.join(pat_name, file_name[2] ) )
            
            if pat<=80:
                for ind_slice in range(0,sizeData[2]):
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[2] ), 
                                       'mask_path': os.path.join(pat_name, file_name[3] ),
                                       'slice': ind_slice } )
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                       'mask_path': os.path.join(pat_name, file_name[5] ),
                                       'slice': ind_slice } )
            else:
                for ind_slice in range(0,sizeData[2]):
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[2] ), 
                                       'mask_path': os.path.join(pat_name, file_name[3] ),
                                       'slice': ind_slice } )    
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                        'mask_path': os.path.join(pat_name, file_name[5] ),
                                        'slice': ind_slice } )
            pat = pat+1
            
    return data_list_tr, data_list_te


def CreateDataset_4D(path_data):      

    data_list_tr = []
    pat = 0

    d = os.listdir(path_data)
    for ii in range(88,89):
        dir_name = d[ii]
    # for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            
            pat_name = os.path.join(path_data, dir_name)
            file_name = os.listdir(pat_name)
            
            sizeData = size_nii( os.path.join(pat_name, file_name[1] ) )
            
            for ind_slice in range(0,sizeData[2]):
            # for ind_slice in range(0,1):
                for t in range(0,sizeData[3]):
                # for t in range(0,1):
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[1] ), 
                                       'slice': ind_slice, 
                                        'time': t,
                                        'Patient': dir_name } )
            pat = pat+1
            
    return data_list_tr



def CreateDatasetOur(path_data ):      

    data_list = []
    for dir_name in os.listdir(path_data):
        if os.path.isdir(os.path.join(path_data, dir_name)):
            sdir_list = os.listdir(os.path.join(path_data, dir_name))
            sdir_list.sort()
             
            for _, sdir_name in enumerate(sdir_list):
                path1 = os.path.join(path_data, dir_name, sdir_name)
                file_list = os.listdir(path1)
                file_list.sort()
                
                for ind_slice, slice_name in enumerate( file_list ):
                
                    dataset = dcm.dcmread( os.path.join( path1, slice_name ) )
                    t = dataset.SeriesDescription  
                    sizeData = size_nii( os.path.join( path1, slice_name ) )
                    
                    if  (np.array(sizeData[0:2])>127).all():
                    #     if t.find('T1')>=0:
                    #         if t.find('post')>=0 or t.find('Post')>=0:
                    #         # if t.find('PRE')>=0 or t.find('Pre')>=0:  
                                
                        if t.find('T2')>=0:
        
                                # ind_slice=0
                                # for slice_name in os.listdir(path1): 
                                    data_list.append( {'img_path': os.path.join(path1, slice_name), 
                                                       'mask_path': 'None',
                                                       'slice': ind_slice,
                                                       'Description': t,
                                                       'Patient': dir_name,
                                                       'Size': sizeData} ) 
                            # ind_slice+=1
            
    # data_list.sort()            
    return data_list



def CreateDataset_StT_J_dcm(path_data):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                # if file.find('Joint')>=0 and file.find('_W4')>=0:
                if file.find('Joint')>=0:
                    path_mask = os.path.join(path_data, pat_name, file)
                    name = file[0:file.find('_gt')] + file[file.find('_gt')+3:]
                    # path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                    path_maps = os.path.join(path_data, pat_name, name)
                    seq = file.split('_')
                # if file.find('_W4')>=0:   
                    # sizeData = Loaders.size( path_maps )
                    sizeData = size_nii( path_maps )

                    if len(sizeData)==2:
                        sizeData = sizeData + (1,)
                    # print(sizeData)
                    
                    for sl in range(0,sizeData[2]):
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'slice': seq[-1][0:3],
                                              'Seq': seq[1],
                                              'ID_pat': ii,
                                              'ID_scan': iii
                                              } )
                    iii+=1
            
    return data_list_tr

def CreateDataset_StT_P_dcm(path_data, text1, text2):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        if pat_name.find(text1)>=0:
            for _,file in enumerate(f):
                if file.find('_gt')>0:
                    # if file.find('Joint')>=0 and file.find('_W')<0:
                    # if file.find(text1)>=0 and
                    if file.find(text2)>=0:
                        
                        path_mask = os.path.join(path_data, pat_name, file)
                        name = file[0:file.find('_gt')] + file[file.find('_gt')+3:]
                        # path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                        path_maps = os.path.join(path_data, pat_name, name)
                        seq = file.split('_')
                    # if file.find('_W4')>=0:   
                        # sizeData = Loaders.size( path_maps )
                        sizeData = size_nii( path_maps )
    
                        if len(sizeData)==2:
                            sizeData = sizeData + (1,)
                        # print(sizeData)
                        
                        for sl in range(0,sizeData[2]):
                            data_list_tr.append( {'img_path': path_maps,
                                                  'mask_path': path_mask,
                                                  'pat_name': pat_name,
                                                  'file_name': name,
                                                  'slice': seq[2],
                                                  'Seq': seq[1],
                                                  'ID_pat': ii,
                                                  'ID_scan': iii
                                                  } )
                        iii+=1
            
    return data_list_tr


def CreateDataset_StT_UnL_dcm(path_data, text1, text2):
    data_list_tr = []
    p = os.listdir(path_data)
    p = sorted(p)
    iii=0
    for ii in range(0,len(p)):
        pat_name = p[ii]
        # if pat_name.find('P')>=0 and not(  int(pat_name[1:-1])<=30 and int(pat_name[1:-1])>=150 ):
        if pat_name.find(text1)>=0:
            pthX = os.path.join(path_data, pat_name)
            if os.path.isdir(pthX):
                f = os.listdir(pthX)
                for _,sf in enumerate(f):
                    pth = os.path.join(path_data, pat_name, sf)
                    if sf.find(text2)>=0:
                        if os.path.isdir(pth):
                            ff = os.listdir(pth)  
                            for sl,file in enumerate(ff):
                                if file.find('.dcm')>=0:
                                    pth2 = os.path.join(path_data, pat_name, sf, file)
                                    data_list_tr.append( {'img_path': pth2,
                                                          'mask_path': pth2,
                                                          'pat_name': pat_name,
                                                          'file_name': sf,
                                                          'slice': sl,
                                                          'Seq': sf[0:2],
                                                          'ID_pat': ii,
                                                          'ID_scan': iii
                                                         } )
                            iii=iii+1 
    return data_list_tr


def CreateDataset_MyoPS_dcm(path_data):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                # if file.find('Joint')>=0 and file.find('_W')<0:
                # if file.find('Joint')>=0:
                    seq = file.split('_')
                    path_mask = os.path.join(path_data, pat_name, file)
                    name = file[0:file.find('_gt')] + file[file.find('_gt')+3:]
                    # path_maps = os.path.join(path_data, pat_name, name+'.nii.gz')
                    path_maps = os.path.join(path_data, pat_name, name)
                    
                # if file.find('_W4')>=0:   
                    # sizeData = Loaders.size( path_maps )
                    sizeData = size_nii( path_maps )

                    if len(sizeData)==2:
                        sizeData = sizeData + (1,)
                    # print(sizeData)
                    
                    for sl in range(0,sizeData[2]):
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'slice': seq[4],
                                              'Seq': seq[3],
                                              'ID_pat': ii,
                                              } )
    return data_list_tr


def CreateDataset( ):  
    #### datasets
    data_list_2=[]
    ## ACDC
    # path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
    # data_list_2_train, data_list_2_test = CreateDataset_ACDC(os.path.normpath( path_data )
                                                             
    ## StT LABELLED - JOINT
    # # path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
    # path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
    # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
    # data_list_2 = data_list_2 + data_list
    
    ## StT LABELLED - P1-30
    path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
    data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
    # b = int(len(data_list)*0.55)
    # random.shuffle(data_list_4_train)
    data_list_2 = data_list_2 + data_list
    
    ## Dataset - MyoPS
    # path_data = '/data/rj21/Data/Data_MyoPS'  # Linux bioeng358
    path_data = '/data/rj21/Data/Data_1mm/MyoPS'  # Linux bioeng358
    data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    data_list_2 = data_list_2+data_list

    
    ## Dataset - EMIDEC
    # path_data = '/data/rj21/Data/Data_emidec'  # Linux bioeng358
    path_data = '/data/rj21/Data/Data_1mm/Emidec'  # Linux bioeng358
    data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    data_list_2 = data_list_2+data_list
    
    ## Dataset - Alinas data
    path_data = '/data/rj21/Data/Data_1mm/T2_alina'  # Linux bioeng358
    data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
    data_list_2 = data_list_2+data_list

    
    # StT UNLABELLED
    path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
    data_list = CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ),'','')
    data_list_3 = data_list


    return data_list_2, data_list_3