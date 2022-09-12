
from enum import Enum

from modules.network.models.DeepVONet import DeepVONet_LSTM, DeepVONet_GRU, DeepVONet_FSM
from modules.network.models.SmallDeepVONet import SmallDeepVONet_LSTM
from modules.network.models.QuaternionSmallDeepVONet import QuaternionSmallDeepVONet_LSTM
from modules.network.models.QuaternionDeepVONet import QuaternionDeepVONet_LSTM, QuaternionDeepVONet_FSM
from modules.network.models.DSC_VONet import DSC_VONet_LSTM
from modules.network.models.QuaternionDSC_VONet import QuaternionDSC_VONet_LSTM

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

    @staticmethod
    def channelsRequired(typeModel):
        if typeModel == ModelEnum.DeepVONet_LSTM:
            return 6
        elif typeModel == ModelEnum.DeepVONet_GRU:
            return 6
        elif typeModel == ModelEnum.SmallDeepVONet_LSTM:
            return 4
        elif typeModel == ModelEnum.DeepVONet_FSM:
            return 6
        elif typeModel == ModelEnum.DSC_VONet_LSTM:
            return 6

        elif typeModel == ModelEnum.QuaternionDeepVONet_LSTM:
            return 8
        elif typeModel == ModelEnum.QuaternionSmallDeepVONet_LSTM:
            return 8
        elif typeModel == ModelEnum.QuaternionDeepVONet_FSM:
            return 8
        elif typeModel == ModelEnum.QuaternionDSC_VONet_LSTM:
            return 8


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
            model = SmallDeepVONet_LSTM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.DeepVONet_FSM:
            model = DeepVONet_FSM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.DSC_VONet_LSTM:
            model = DSC_VONet_LSTM(input_size, hidden_size).to(device)

        elif typeModel == ModelEnum.QuaternionDeepVONet_LSTM:
            model = QuaternionDeepVONet_LSTM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionSmallDeepVONet_LSTM:
            model = QuaternionSmallDeepVONet_LSTM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionDeepVONet_FSM:
            model = QuaternionDeepVONet_FSM(input_size, hidden_size).to(device)
        elif typeModel == ModelEnum.QuaternionDSC_VONet_LSTM:
            model = QuaternionDSC_VONet_LSTM(input_size, hidden_size).to(device)

        else:
            PM.printI(bcolors.LIGHTRED+"ERROR {}".format(typeModel)+bcolors.ENDC+"\n")
            raise ValueError

        return model




