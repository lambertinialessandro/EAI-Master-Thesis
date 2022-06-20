
from enum import Enum

from modules.network.models.DeepVONet import DeepVONet, DeepVONet_FSM
from modules.network.models.SmallDeepVONet import SmallDeepVONet
from modules.network.models.QuaternionSmallDeepVONet import QuaternionSmallDeepVONet
from modules.network.models.QuaternionDeepVONet import QuaternionDeepVONet, QuaternionDeepVONet_FSM


class ModelEnum(Enum):
    DeepVONet = "DeepVONet"
    QuaternionDeepVONet = "QuaternionDeepVONet"

    SmallDeepVONet = "SmallDeepVONet"
    QuaternionSmallDeepVONet = "QuaternionSmallDeepVONet"

    DeepVONet_FSM = "DeepVONet_FSM"
    QuaternionDeepVONet_FSM = "QuaternionDeepVONet_FSM"


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def build(typeModel: ModelEnum, input_size_LSTM, hidden_size_LSTM, device):
        if typeModel == ModelEnum.DeepVONet:
            model = DeepVONet(input_size_LSTM, hidden_size_LSTM).to(device)
        elif typeModel == ModelEnum.QuaternionDeepVONet:
            model = QuaternionDeepVONet(input_size_LSTM, hidden_size_LSTM).to(device)

        elif typeModel == ModelEnum.SmallDeepVONet:
            model = SmallDeepVONet(input_size_LSTM, hidden_size_LSTM).to(device)
        elif typeModel == ModelEnum.QuaternionSmallDeepVONet:
            model = QuaternionSmallDeepVONet(input_size_LSTM, hidden_size_LSTM).to(device)

        elif typeModel == ModelEnum.DeepVONet_FSM:
            model = DeepVONet_FSM(input_size_LSTM, hidden_size_LSTM).to(device)
        elif typeModel == ModelEnum.QuaternionDeepVONet_FSM:
            model = QuaternionDeepVONet_FSM().to(device)
        else:
            raise ValueError

        return model

