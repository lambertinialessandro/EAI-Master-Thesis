# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:29:52 2022

@author: lambe
"""

#@title {vertical-output: true, form-width: "50%", display-mode: "code"}
#@markdown # Global Variables:

#@markdown ---
#@markdown ### Flags:
FLAG_DOWNLOAD_DATASET = False #@param {type:"boolean"}
FLAG_DEBUG_PRINT = True #@param {type:"boolean"}
FLAG_INFO_PRINT = True #@param {type:"boolean"}

#@markdown ---
#@markdown ### Files path:
# global variables to save the tables/models 
dir_main = './'#@param {type:"string"}

dir_Dataset = 'Dataset/'#@param {type:"string"}
dir_Dataset = dir_main + dir_Dataset
path_sequences = dir_Dataset+'sequences/'
path_poses = dir_Dataset+'poses/'

dir_Model = 'Model/'#@param {type:"string"}
dir_Model = dir_main + dir_Model
dir_History = 'History/'#@param {type:"string"}
dir_History = dir_main + dir_History


#@markdown ---
#@markdown ### Images settings:
BACH_SIZE = 10 #@param {type:"number"}
CHANNELS = 6
WIDTH = 320 #@param [320, 640, 1280] {type:"raw", allow-input: false}
HEIGHT = 96 #@param[96, 192, 384] {type:"raw", allow-input: false}
NUM_POSES = 6

img_size = (320, 96) # (1280,384) # (640, 192) # (320, 96)

trainingSeries = ["00", "02", "08", "09"]
testingSeries = ["03", "04", "05", "06", "07", "10"]



