
from enum import Enum

from modules.network.models.DeepVONet import DeepVONet_LSTM, DeepVONet_GRU, DeepVONet_FSM
from modules.network.models.SmallDeepVONet import SmallDeepVONet
from modules.network.models.QuaternionSmallDeepVONet import QuaternionSmallDeepVONet
from modules.network.models.QuaternionDeepVONet import QuaternionDeepVONet, QuaternionDeepVONet_FSM
from modules.network.models.DSC_VONet import DSC_VONet
from modules.network.models.QuaternionDSC_VONet import QuaternionDSC_VONet

from modules.utility import PM, bcolors


class ModelEnum(Enum):
    DeepVONet_LSTM = "DeepVONet_LSTM"
    DeepVONet_GRU = "DeepVONet_GRU"
    SmallDeepVONet_LSTM = "SmallDeepVONet_LSTM"
    DeepVONet_FSM = "DeepVONet_FSM"
    DSC_VONet_LSTM = "DSC_VONet_LSTM"

    QuaternionDeepVONet_LSTM = "QuaternionDeepVONet_LSTM"
    QuaternionSmallDeepVONet_LSTM = "QuaternionSmallDeepVONet_LSTM"
    QuaternionDeepVONet_FSM = "QuaternionDeepVONet_FSM"
    QuaternionDSC_VONet_LSTM = "QuaternionDSC_VONet_LSTM"


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def build(typeModel: ModelEnum, input_size, hidden_size, device):
        if typeModel == ModelEnum.DeepVONet_LSTM:
            model = DeepVONet_LSTM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.DeepVONet_GRU:
            model = DeepVONet_GRU(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.SmallDeepVONet_LSTM:
            model = SmallDeepVONet(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.DeepVONet_FSM:
            model = DeepVONet_FSM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.DSC_VONet_LSTM:
            model = DSC_VONet(input_size, hidden_size).to(device)

        elif typeModel == ModelEnum.QuaternionDeepVONet_LSTM:
            model = QuaternionDeepVONet(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionSmallDeepVONet_LSTM:
            model = QuaternionSmallDeepVONet(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionDeepVONet_FSM:
            model = QuaternionDeepVONet_FSM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionDSC_VONet_LSTM:
            model = QuaternionDSC_VONet(input_size, hidden_size).to(device)

        else:
            PM.printI(bcolors.LIGHTRED+"ERROR {}".format(typeModel)+bcolors.ENDC+"\n")
            raise ValueError

        return model


if __name__ == "__main__":
    import torch
    import math

    WIDTH = 320
    HEIGHT = 96

    DIM_LSTM = 10240 # 384 * math.ceil(WIDTH/2**6) * math.ceil(HEIGHT/2**6)
    HIDDEN_SIZE_LSTM = 1000



    #x = torch.rand(5, 6, WIDTH, HEIGHT)
    #out_m1 = m1(x)
    #print(out_m1.shape)

    m1 = ModelFactory.build(ModelEnum.QuaternionDSC_VONet,
                            DIM_LSTM, HIDDEN_SIZE_LSTM, torch.device("cpu"))
    params_m1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"The standard convolution uses {params_m1} parameters.")



    # ------------------------ #
    # LSTM 1000   52.976.000
    # LSTM 100     4.212.200
    # ------------------------ #


    # NO LSTM      LSTM 100     LSTM 1000
    # 14.626.102   18.838.302   67.602.102   DeepVONet
    #  3.665.694    7.883.294   56.647.094   QuaternionDeepVONet

    #  1.596.898    5.809.098   54.572.898   DSC_VONet
    #    435.526    4.647.726   53.411.526   QuaternionDSC_VONet

    # 27.760.442   -            -            DeepVONet_FSM
    # 10.275.998   -            -            QuaternionDeepVONet_FSM







