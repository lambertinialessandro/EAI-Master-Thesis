
#@title  { run: "auto", vertical-output: true, form-width: "50%", display-mode: "both" }
#@markdown # Global Variables:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import torch
from enum import Enum

from modules.network.ModelModule import ModelEnum
from modules.network.CriterionModule import CriterionEnum
from modules.network.OptimizerModule import OptimizerEnum

from modules.preprocess.PreprocessModule import PreprocessEnum, PreprocessFactory


# from params import ParamsInstance as params

#from trainModel import trainEnum

class trainEnum(Enum):
    preprocessed = "preprocessed"
    online = "online"

    preprocessed_random = "preprocessed_random"
    online_random = "online_random"

    preprocessed_random_RDG = "preprocessed_random_RDG"
    online_random_RDG = "online_random_RDG"


class imageSizeEnum(Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    BIG = "BIG"

class Params:
    _instance = None

    ModelEnum = ModelEnum
    CriterionEnum = CriterionEnum
    OptimizerEnum = OptimizerEnum

    PreprocessEnum = PreprocessEnum

    imageSizeEnum = imageSizeEnum

    trainEnum = trainEnum


    def __new__(cls, *args, **kwargs):
        if Params._instance is None:
            Params._instance = super(Params, cls).__new__(cls)
        return Params._instance

    def __init__(self):
        self.dir_main = '.'
        self._dir_Dataset = 'dataset'
        self._dir_Model = 'Model'
        self._dir_History = 'History'

        self.set_dir_main(self.dir_main)

        self.BACH_SIZE = 5
        self.STEP = 5 #self.setPreprocesF(self.typePreprocess) ############

        self.HIDDEN_SIZE = 1000
        self.NUM_POSES = 6

        self.BASE_EPOCH = 1
        self.END_EPOCH = 200
        self.NUM_EPOCHS = self.END_EPOCH - self.BASE_EPOCH

        self.setImageSize(imageSizeEnum.SMALL) # BIG

        self.setTypeModel(ModelEnum.DeepVONet_LSTM)
        self.setTypeCriterion(CriterionEnum.MSELoss)
        self.setTypeOptimizer(OptimizerEnum.Adam)

        self.setPreprocesF(PreprocessEnum.RESIZED)

        self.setTypeTrain(trainEnum.online_random_RDG)


        # ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        self.trainingSeries = ["00", "01", "02", "08", "09"] # [4541, 1101, 4661, 4071, 1591]
        self.testingSeries = ["03", "04", "05", "06", "07", "10"]


        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")


        self.prefixFileNameLoad = "DeepVO_epoch_"

        self.prefixFileNameLosses = "loss_"
        self.suffixFileNameLosses = "{}[{}]"

        self.prefixFileNameSave = "DeepVO_epoch_"
        self.suffixFileNameSave = "{}[{}-{}]"
        self.setSaveLoadModelParams("", False, "", True, True, 35)


    def set_dir_main(self, dir_main):
        self.dir_main = dir_main

        self.dir_Dataset = os.path.join(self.dir_main, self._dir_Dataset)
        self.path_sequences = os.path.join(self.dir_Dataset, 'sequences')
        self.path_poses = os.path.join(self.dir_Dataset, 'poses')

        self.dir_Model = os.path.join(self.dir_main, self._dir_Model)
        self.dir_History = os.path.join(self.dir_main, self._dir_History)

    def set_dir_Dataset(self, dir_Dataset):
        self._dir_Dataset = dir_Dataset

        self.dir_Dataset = os.path.join(self.dir_main, self._dir_Dataset)
        self.path_sequences = os.path.join(dir_Dataset, 'sequences')
        self.path_poses = os.path.join(self.dir_Dataset, 'poses')

    def setTypeModel(self, typeModel: ModelEnum):
        self.typeModel = typeModel

        self.CHANNELS = ModelEnum.channelsRequired(self.typeModel)


        if self.CHANNELS == 4:
            self.DIM_RNN = 384 * math.ceil(self.WIDTH/2**6) * math.ceil(self.HEIGHT/2**6)
        elif self.CHANNELS == 6:
            self.DIM_RNN = 1024 * math.ceil(self.WIDTH/2**6) * math.ceil(self.HEIGHT/2**6)
        elif self.CHANNELS == 8:
            self.DIM_RNN = 1024 * math.ceil(self.WIDTH/2**6) * math.ceil(self.HEIGHT/2**6)
        else:
            raise ValueError

        #self.setPreprocesF(self.typePreprocess) ############

    def setTypeCriterion(self, typeCriterion: CriterionEnum):
        self.typeCriterion = typeCriterion

    def setTypeOptimizer(self, typeOptimizer: OptimizerEnum):
        self.typeOptimizer = typeOptimizer

    def setImageSize(self, typeImg: imageSizeEnum):
        if typeImg == imageSizeEnum.SMALL:
            self.WIDTH = 320
            self.HEIGHT = 96
        elif typeImg == imageSizeEnum.MEDIUM:
            self.WIDTH = 640
            self.HEIGHT = 192
        elif typeImg == imageSizeEnum.BIG:
            self.WIDTH = 1280
            self.HEIGHT = 384
        else:
            raise ValueError

        self.img_size = (self.WIDTH, self.HEIGHT)

        #self.setPreprocesF(self.typePreprocess) ############

    def setPreprocesF(self, typePreprocess: PreprocessEnum):
        appPreprocesF = PreprocessFactory.build(typePreprocess, self.img_size, step=self.STEP)
        if(appPreprocesF.ch*2 == self.CHANNELS):
            self.typePreprocess = typePreprocess
            self.preprocesF = appPreprocesF
            self.suffixType = self.preprocesF.suffix()
        else:
            raise ValueError

    def setBaseEpoch(self, base_epoch):
        if(base_epoch < self.END_EPOCH):
            self.BASE_EPOCH = base_epoch
            self.NUM_EPOCHS = self.END_EPOCH - self.BASE_EPOCH
        else:
            raise ValueError

    def setEndEpoch(self, end_epoch):
        if(self.BASE_EPOCH < end_epoch):
            self.END_EPOCH = end_epoch
            self.NUM_EPOCHS = self.END_EPOCH - self.BASE_EPOCH
        else:
            raise ValueError

    def setTypeTrain(self, typeTrain: trainEnum):
        self.type_train = typeTrain

    def setSaveLoadModelParams(self, fileNameFormat,  FLAG_LOAD, suffixFileNameLoad,
                               FLAG_SAVE_LOG,  FLAG_SAVE, SAVE_STEP):
        self.fileNameFormat = fileNameFormat

        self.FLAG_LOAD = FLAG_LOAD
        self.suffixFileNameLoad = suffixFileNameLoad # ex: "medium[11-18]"
        self.FLAG_SAVE_LOG = FLAG_SAVE_LOG
        self.FLAG_SAVE = FLAG_SAVE
        self.SAVE_STEP = SAVE_STEP

ParamsInstance = Params()

