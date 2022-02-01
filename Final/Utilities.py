import numpy as np
# import torch
import os
# import random
# import xlsxwriter
# import pandas as pd
# import pydicom as dcm
# import torchvision.transforms as T
# import cv2

import Loaders

def crop_min(img):
    
    s = min( img.shape)
    img = img[0:s,0:s]
    
    return img

def crop_center(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    new_width_old = new_width
    new_heigh_old = new_height
    
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    new_width = min(width, new_width)
    new_height = min(height, new_height)
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2] 
        
    padNUm=[] 
    padNUm.append(int(np.floor((new_width_old-center_cropped_img.shape[0])/2)))
    padNUm.append(int(np.ceil((new_width_old-center_cropped_img.shape[0])/2)))
    padNUm.append(int(np.floor((new_heigh_old-center_cropped_img.shape[1])/2)))
    padNUm.append(int(np.ceil((new_heigh_old-center_cropped_img.shape[1])/2)))
    padNUm = tuple(padNUm)
    
    center_cropped_img = np.pad(center_cropped_img, [padNUm[0:2],padNUm[2:4]], mode='constant', constant_values=(0, 0))
        
    return center_cropped_img, (top, bottom, left, right), padNUm

