#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:49:44 2022

@author: rj21
"""
import Utilities as Util
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dcm


# np.random.seed(1)
# x = np.random.rand(20,20)

path_save = '/data/rj21/MyoSeg/Export/contrast_augment'

img_path = '/data/rj21/Data/Data_StT_Unlabeled/A025_/T1-preGd-contrasts_series0026-Body/img0006-49.1244.dcm'

dataset = dcm.dcmread(img_path)
img = dataset.pixel_array.astype(dtype='float32')
img = img[50:125,40:120]

plt.Figure()
plt.imshow(img)
plt.show()

phis = np.arange(0,np.pi*2,0.5)

for _,i in enumerate(phis):
    params = [0.2,3, i ]
    y = Util.random_contrast(img,params)

    plt.Figure()
    plt.imshow(y)
    Fig = plt.gcf()
    plt.show()
    plt.draw()
    Fig.savefig( path_save + '/' + 'augm_' +  str(i)  + '.png', dpi=150)
    plt.close(Fig)
    
    t = np.arange(0,1,0.01)
    plt.Figure()
    plt.plot( t+ params[0]*np.cos( params[1]*2*np.pi*t + params[2]))
    Fig = plt.gcf()
    plt.show()
    plt.draw()
    Fig.savefig( path_save + '/' + 'trans_' +  str(i)  + '.png', dpi=150)
    plt.close(Fig)
