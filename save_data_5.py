#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:37:49 2022

@author: rj21
"""

## pro jen nektere database, jako predtim u MyoSeg

from file_folder_utils import subdirs, subfiles, prepare_nnUNet_file_structure, generate_dataset_json_new
import os
import pandas as pd
from datetime import datetime
import glob
import string
import random
import nibabel as nib
import matplotlib as plt

## resaving/create xlas file for MyoSeg Unet from nifti, the same way as for nnUNet

path = r'/data/rj21/Data/Data_all_nii'
outpath = r'/data/rj21/MyoSeg/params_net'

Task = 'Task743_MyoSeg'

datatsets_list = subdirs(path)
# patients_list = glob.glob(path+os.sep+'**'+os.sep+'Data.nii.gz', recursive=True)


# prepare the list of all patients and series
scans_list = pd.DataFrame({'Dataset' : []})
scans_list.loc[0, ('Patient')] = ''
scans_list.loc[0, ('Series')] = ''
scans_list.loc[0, ('Data_path')] = ''

random.seed(77)
i=-1
for k, dat in enumerate(datatsets_list):
    
    patients_list = glob.glob(dat +os.sep+'**'+os.sep+'Data.nii.gz', recursive=True)
    random.shuffle(patients_list)
    
    for l, file in enumerate(patients_list):
        vel = nib.load(file).shape
        for sl in range(vel[2]):
            i=i+1
            f = file.split(os.sep)
            scans_list.loc[i, ('Dataset')] = os.path.basename(dat)
            scans_list.loc[i, ('Patient')] = f[6]
            scans_list.loc[i, ('Series')] = os.sep.join(f[6:-1])
            scans_list.loc[i, ('Slice')] = sl
            scans_list.loc[i, ('Data_path')] = file
            
            if k>=5:
                scans_list.loc[i, ('Set')] = 'train'
            else:
                scans_list.loc[i, ('Set')] = 'test'
        
scans_list.sort_values(by=['Patient'])

# scans_list.loc[0:600, ('Set')] = 'train'
# scans_list.loc[600:len(scans_list), ('Set')] = 'test'

scans_list.to_excel(outpath+os.sep+"scans_list_"+Task+".xlsx")
