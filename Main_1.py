import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils import data
import torch.optim as optim
import glob

import Utilities as Util
import Unet_2D


path_data = '/data/rj21/MyoSeg/Data_ACDC/training'

data_list_train, data_list_test = Util.CreateDataset(path_data)

