
#@title  { run: "auto", vertical-output: true, form-width: "50%", display-mode: "both" }
#@markdown # Global Variables:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import torch

from modules.utility import PM

#@markdown ---
#@markdown ### Flags:
FLAG_DOWNLOAD_DATASET = False #@param {type:"boolean"}
FLAG_DEBUG_PRINT = True #@param {type:"boolean"}
FLAG_INFO_PRINT = True #@param {type:"boolean"}
FLAG_OUT_HTML = False #@param {type:"boolean"}
PM.setFlags(FLAG_DEBUG_PRINT, FLAG_INFO_PRINT, FLAG_OUT_HTML)

#@markdown ---
#@markdown ### Files path:
# global variables to save the tables/models
dir_main = '.'#@param {type:"string"}

#dir_main = 'drive/.shortcut-targets-by-id/1u8wbljmLaX2INDIFTQqsk3xLVCalv2_o/Thesis/'#@param {type:"string"}
#FLAG_OUT_HTML = True #@param {type:"boolean"}

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
    # TODO
typeModel = "DeepVONet" #@param ["DeepVONet", "DeepVONet_FSM",
                        #        "QuaternionDeepVONet", "QuaternionDeepVONet_FSM"] {type:"string"}
typeCriterion = "MSELoss" #@param ["MSELoss"] {type:"string"}
typeOptimizer = "Adam" #@param ["Adam", "SGD"] {type:"string"}

#@markdown ---
#@markdown ### Images settings:
BACH_SIZE = 10 #@param {type:"number"}

if typeModel == "DeepVONet":
    CHANNELS = 6
elif typeModel == "QuaternionDeepVONet":
    CHANNELS = 8
else:
    raise ValueError

    # TODO
suffixType = "SOBEL" # UNCHANGED, SOBEL, CROPPING, QUAT_PURE, QUAT_GRAY, QUAT_CED

BACH_SIZE = 10 #@param {type:"number"}
NUM_BACH = 4 #@param {type:"number"} # = NUM_BACH * BACH_SIZE
RDG_ITER = 1 #@param {type:"number"}
STEP = 5 # 5 #@param {type:"number"}

#@param [320, 640, 1280] {type:"raw", allow-input: false}
#@param[96, 192, 384] {type:"raw", allow-input: false}
#@param [6, 8] {type:"raw", allow-input: false}

__dim_image = 3
if __dim_image == 1:
    DIM_LSTM = 3840 # 3840 = 384 * 10        10240 # = 1024 * 10
    WIDTH = 320
    HEIGHT = 96

    # C = 1024 # TODO
    # batch, dim1, dim2, channels = 10, 5, 2, 128 # TODO

elif __dim_image == 2:
    DIM_LSTM = 30720 # = 3 * 10240
    WIDTH = 640
    HEIGHT = 192
elif __dim_image == 3:
    DIM_LSTM = 122880 # = 12 * 10240 || = 4 * 30720
    WIDTH = 1280
    HEIGHT = 384
else:
    raise ValueError
#DIM_LSTM = 1024 * math.ceil(WIDTH/2**6) * math.ceil(HEIGHT/2**6)
HIDDEN_SIZE_LSTM = 1000

NUM_POSES = 6

img_size = (WIDTH, HEIGHT) # (1280,384) # (640, 192) # (320, 96)

# [4541, 1101, 4661, 4071, 1591]
trainingSeries = ["00", "01", "02", "08", "09"] # added 01
# ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
testingSeries = ["03", "04", "05", "06", "07", "10"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



FLAG_LOAD = False #@param {type:"boolean"}
FLAG_SAVE_LOG = True #@param {type:"boolean"}
SAVE_STEP = 35 #@param {type:"number"}
FLAG_SAVE = True #@param {type:"boolean"}

BASE_EPOCH = 1 #@param {type:"number"} # 1 starting epoch
NUM_EPOCHS = 200 - BASE_EPOCH #@param {type:"number"} # 10 how many epoch

fileNameFormat = "medium"

prefixFileNameLoad = "DeepVO_epoch_"
suffixFileNameLoad = "medium[11-18]" #@param {type:"string"}

prefixFileNameLosses = "loss_"
suffixFileNameLosses = "{}[{}]"

prefixFileNameSave = "DeepVO_epoch_"
suffixFileNameSave = "{}[{}-{}]"

type_train = "preprocessed"



