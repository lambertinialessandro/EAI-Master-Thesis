# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:29:52 2022

@author: lambe
"""

#@title  { run: "auto", vertical-output: true, form-width: "50%", display-mode: "both" }
#@markdown # Global Variables:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch

#@markdown ---
#@markdown ### Flags:
FLAG_DOWNLOAD_DATASET = False #@param {type:"boolean"}
FLAG_DEBUG_PRINT = True #@param {type:"boolean"}
FLAG_INFO_PRINT = True #@param {type:"boolean"}

#@markdown ---
#@markdown ### Files path:
# global variables to save the tables/models
dir_main = '.'#@param {type:"string"}

dir_Dataset = 'Dataset'#@param {type:"string"}
dir_Dataset = os.path.join(dir_main, dir_Dataset)
path_sequences = os.path.join(dir_Dataset, 'sequences')
path_poses = os.path.join(dir_Dataset, 'poses')

dir_Model = 'Model'#@param {type:"string"}
dir_Model = os.path.join(dir_main, dir_Model)
dir_History = 'History'#@param {type:"string"}
dir_History = os.path.join(dir_main, dir_History)

#@markdown ---
#@markdown ### Model settings:
typeModel = "DeepVONet" #@param ["DeepVONet", "QuaternionDeepVONet"] {type:"string"}
typeCriterion = "MSELoss" #@param ["MSELoss"] {type:"string"}
typeOptimizer = "Adam" #@param ["Adam", "SGD"] {type:"string"}

#@markdown ---
#@markdown ### Images settings:
BACH_SIZE = 10 #@param {type:"number"}

if typeModel == "DeepVONet":
    CHANNELS = 6
    suffixType = 1
elif typeModel == "QuaternionDeepVONet":
    CHANNELS = 8
    suffixType = 2
else:
    raise ValueError

BACH_SIZE = 10 #@param {type:"number"}
NUM_BACH = 1 #@param {type:"number"} # = NUM_BACH * BACH_SIZE

#@param [320, 640, 1280] {type:"raw", allow-input: false}
#@param[96, 192, 384] {type:"raw", allow-input: false}
#@param [6, 8] {type:"raw", allow-input: false}

__dim_image = 1
if __dim_image == 1:
    DIM_LSTM = 10240 # = 1024 * 10
    WIDTH = 320
    HEIGHT = 96
elif __dim_image == 2:
    DIM_LSTM = 307200 # = 30 * 10240
    WIDTH = 320
    HEIGHT = 96
elif __dim_image == 3:
    DIM_LSTM = 1228800 # = 120 * 10240 || = 4 * 307200
    WIDTH = 320
    HEIGHT = 96
else:
    raise ValueError

NUM_POSES = 6

img_size = (320, 96) # (1280,384) # (640, 192) # (320, 96)

trainingSeries = ["00", "01", "02", "08", "09"] # added 01
testingSeries = ["03", "04", "05", "06", "07", "10"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



