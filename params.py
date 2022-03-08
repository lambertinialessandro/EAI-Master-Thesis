# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:29:52 2022

@author: lambe
"""

FLAG_DOWNLOAD_DATASET = False #@param {type:"boolean"}
FLAG_DEBUG_PRINT = True #@param {type:"boolean"}
FLAG_INFO_PRINT = True #@param {type:"boolean"}

# global variables to save the tables/models
dir_main = './'#@param {type:"string"}

dir_Dataset = 'Dataset/'#@param {type:"string"}
dir_Model = 'Model/'#@param {type:"string"}
dir_History = 'History/'#@param {type:"string"}

dir_Dataset = dir_main + dir_Dataset
path_sequences = dir_Dataset+'sequences/'
path_poses = dir_Dataset+'poses/'
dir_Model = dir_main + dir_Model
dir_History = dir_main + dir_History

img_size = (640, 174) # (1280,384) # (640, 174) = 111.360