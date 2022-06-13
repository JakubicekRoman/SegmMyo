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
                                       'slice': ind_slice,
                                       'pat_name': os.path.basename(pat_name),
                                       'file_name': os.path.basename(file_name[2])[0:-7:1],
                                       'Seq': 'cine',
                                       'Dts': 'ACDC',
                                       'Type': ''} )
                    data_list_tr.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                       'mask_path': os.path.join(pat_name, file_name[5] ),
                                       'slice': ind_slice,
                                       'pat_name': os.path.basename(pat_name),
                                       'file_name': os.path.basename(file_name[4])[0:-7:1],
                                       'Seq': 'cine',
                                       'Dts': 'ACDC',
                                       'Type': ''} )
            else:
                for ind_slice in range(0,sizeData[2]):
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[2] ), 
                                       'mask_path': os.path.join(pat_name, file_name[3] ),
                                       'slice': ind_slice,
                                       'pat_name': os.path.basename(pat_name),
                                       'file_name': os.path.basename(file_name[2])[0:-7:1],
                                       'Seq': 'cine',
                                       'Dts': 'ACDC',
                                       'Type': ''} )   
                    data_list_te.append( {'img_path': os.path.join(pat_name, file_name[4] ), 
                                        'mask_path': os.path.join(pat_name, file_name[5] ),
                                        'slice': ind_slice,
                                        'pat_name': os.path.basename(pat_name),
                                        'file_name': os.path.basename(file_name[4])[0:-7:1],
                                        'Seq': 'cine',
                                        'Dts': 'ACDC',
                                        'Type': ''} )
            pat = pat+1
            
    return data_list_tr, data_list_te



def CreateDataset_StT_J_dcm(path_data, text):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                if file.find('Joint')>=0 and file.find(text)>=0:
                # if file.find('Joint')>=0:
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
                                              'Dts': seq[0],
                                              'Type': ''
                                              # 'ID_pat': ii,
                                              # 'ID_scan': iii
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
                                                  'Seq': seq[0][0:2],
                                                  'Dts': 'Clin_StT',
                                                  'Type': seq[1]
                                                  # 'ID_pat': ii,
                                                  # 'ID_scan': iii
                                                  } )
                        iii+=1
            
    return data_list_tr

def CreateDataset_StT_Alinas_dcm(path_data, text1, text2):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        if pat_name.find(text1)>=0:
            for _,file in enumerate(f):
                if file.find('_gt')>0:
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
                                                  'Seq': seq[0],
                                                  'Dts': '3D_T2',
                                                  'Type': seq[1]
                                                  # 'ID_pat': ii,
                                                  # 'ID_scan': iii
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
        if pat_name.find('P')>=0 and not(  int(pat_name[1:-1])<=30 and int(pat_name[1:-1])>=150 ):
        # if pat_name.find(text1)>=0:
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


def CreateDataset_MyoPS_dcm(path_data,text1):
    data_list_tr = []
    iii=0
    p = os.listdir(path_data)
    for ii in range(0,len(p)):
        pat_name = p[ii]
        f = os.listdir(os.path.join(path_data, pat_name))
        for _,file in enumerate(f):
            if file.find('_gt')>0:
                # if file.find('Joint')>=0 and file.find('_W')<0:
                if file.find(text1)>=0:
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
                        name_seq = seq[3].replace('C0','cine')
                        name_seq = name_seq.replace('DE','LGE')
                        name_seq = name_seq.replace('T2','T2')
                        
                        data_list_tr.append( {'img_path': path_maps,
                                              'mask_path': path_mask,
                                              'pat_name': pat_name,
                                              'file_name': name,
                                              'slice': seq[4],
                                              'Seq': name_seq,
                                              'Dts': 'MyoPS',
                                              'Type': ''
                                              # 'ID_pat': ii,
                                              } )
    return data_list_tr

def CreateDataset_Emidec_dcm(path_data):
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
                                              'slice': seq[3],
                                              'Seq': 'T1',
                                              'Dts': 'Emidec',
                                              'Type': seq[1][0]
                                              # 'ID_pat': ii,
                                              } )
    return data_list_tr


def CreateDataset( ):  
    #### datasets
    data_list_2=[]
    ## ACDC
    # path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
    # data_list_2_train, data_list_2_test = CreateDataset_ACDC(os.path.normpath( path_data )
                                                             
    # ## StT LABELLED - JOINT
    # # path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
    # path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
    # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data ))
    # data_list_2 = data_list_2 + data_list
    
    # ## StT LABELLED - P1-30
    # path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
    # data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','_m')
    # # b = int(len(data_list)*0.55)
    # # random.shuffle(data_list_4_train)
    # data_list_2 = data_list_2 + data_list
    
    # ## Dataset - MyoPS
    # # path_data = '/data/rj21/Data/Data_MyoPS'  # Linux bioeng358
    # path_data = '/data/rj21/Data/Data_1mm/MyoPS'  # Linux bioeng358
    # data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    # data_list_2 = data_list_2+data_list

    
    # ## Dataset - EMIDEC
    # # path_data = '/data/rj21/Data/Data_emidec'  # Linux bioeng358
    # path_data = '/data/rj21/Data/Data_1mm/Emidec'  # Linux bioeng358
    # data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ))
    # data_list_2 = data_list_2+data_list
    
    # ## Dataset - Alinas data
    # path_data = '/data/rj21/Data/Data_1mm/T2_alina'  # Linux bioeng358
    # data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','')
    # data_list_2 = data_list_2+data_list

    
    # StT UNLABELLED
    path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
    data_list = CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ),'','')
    data_list_3 = data_list


    return data_list_2, data_list_3

def CreateDataset_div( ):  
    #### datasets
    data_list_2_train=[]
    data_list_2_test=[]
    
    # # ACDC
    path_data = '/data/rj21/Data/Data_ACDC/training'  # Linux bioeng358
    data_list_train, data_list_test = CreateDataset_ACDC(os.path.normpath( path_data ))
    data_list_2_train = data_list_2_train + data_list_train
    data_list_2_test = data_list_2_test + data_list_test

    # # ## StT LABELLED - JOINT
    # # path_data = '/data/rj21/Data/Data_Joint_StT_Labelled/Resaved_data_StT_cropped'  # Linux bioeng358
    path_data = '/data/rj21/Data/Data_1mm/Joint'  # Linux bioeng358
    data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data) , '')    # ## all joint
    # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data) , 'T1')
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    
    # # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data) , 'T2')
    # # b = int(len(data_list)*0.80)
    # # data_list_2_train = data_list_2_train + data_list[1:b]
    # # data_list_2_test = data_list_2_test + data_list[b:-1]
    # # # # data_list_2 = data_list_2 + data_list
    
    # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data) , 'W1')
    # b = int(len(data_list)*0.80)
    # data_list_2_train = data_list_2_train + data_list[1:b]
    # data_list_2_test = data_list_2_test + data_list[b:-1]
    
    # # data_list = CreateDataset_StT_J_dcm(os.path.normpath( path_data) , 'W4')
    # # b = int(len(data_list)*0.80)
    # # data_list_2_train = data_list_2_train + data_list[1:b]
    # # data_list_2_test = data_list_2_test + data_list[b:-1]
    
    
    # # ## StT LABELLED - P1-30 for _m
    path_data = '/data/rj21/Data/Data_StT_Labaled'  # Linux bioeng358
    data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','_m')
    
    # data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','T1pre_m')
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    
    # data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','T2_m')
    # b = int(len(data_list)*0.80)
    # data_list_2_train = data_list_2_train + data_list[1:b]
    # data_list_2_test = data_list_2_test + data_list[b:-1]
    
    data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','T1pre_c1')
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    
    data_list = CreateDataset_StT_P_dcm(os.path.normpath( path_data ),'','T2_c3')
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    
    
        
    # ## Dataset - MyoPS
    # # path_data = '/data/rj21/Data/Data_MyoPS'  # Linux bioeng358
    path_data = '/data/rj21/Data/Data_1mm/MyoPS'  # Linux bioeng358
    
    data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ), '_C0' )
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    
    # data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ), '_DE' )
    # b = int(len(data_list)*0.80)
    # data_list_2_train = data_list_2_train + data_list[1:b]
    # data_list_2_test = data_list_2_test + data_list[b:-1]
    
    data_list = CreateDataset_MyoPS_dcm(os.path.normpath( path_data ), '_T2' )
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]

    
    # # ## Dataset - EMIDEC
    # path_data = '/data/rj21/Data/Data_emidec'  # Linux bioeng358
    path_data = '/data/rj21/Data/Data_1mm/Emidec'  # Linux bioeng358
    data_list = CreateDataset_Emidec_dcm(os.path.normpath( path_data ))
    b = int(len(data_list)*0.80)
    data_list_2_train = data_list_2_train + data_list[1:b]
    data_list_2_test = data_list_2_test + data_list[b:-1]
    # data_list_2 = data_list_2+data_list
    
    # ## Dataset - Alinas data
    # # path_data = '/data/rj21/Data/Data_1mm/T2_alina'  # Linux bioeng358
    # # data_list = CreateDataset_StT_Alinas_dcm(os.path.normpath( path_data ),'','')
    # # b = int(len(data_list)*0.80)
    # # data_list_2_train = data_list_2_train + data_list[1:b]
    # # data_list_2_test = data_list_2_test + data_list[b:-1]
    # # # data_list_2 = data_list_2+data_list

    for i in range(len(data_list_2_train)):
        data_list_2_train[i]['Set']='Train'
    for i in range(len(data_list_2_test)):
        data_list_2_test[i]['Set']='Test'
          
    
    # StT UNLABELLED
    path_data = '/data/rj21/Data/Data_StT_Unlabeled'  # Linux bioeng358
    data_list = CreateDataset_StT_UnL_dcm(os.path.normpath( path_data ),'','')
    data_list_3 = data_list


    return data_list_2_train, data_list_2_test,  data_list_3