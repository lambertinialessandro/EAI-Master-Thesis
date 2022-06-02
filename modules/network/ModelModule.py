
from enum import Enum

import params

from modules.network.DeepVONet import DeepVONet, DeepVONet_FSM
from modules.network.SmallDeepVONet import SmallDeepVONet
from modules.network.QuaternionDeepVONet import QuaternionDeepVONet, QuaternionDeepVONet_FSM


class ModelEnum(Enum):
    DeepVONet = "DeepVONet"
    DeepVONet_FSM = "DeepVONet_FSM"

    SmallDeepVONet = "SmallDeepVONet"

    QuaternionDeepVONet = "QuaternionDeepVONet"
    QuaternionDeepVONet_FSM = "QuaternionDeepVONet_FSM"

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def build(typeModel: ModelEnum):
        if typeModel == ModelEnum.DeepVONet:
            model = DeepVONet().to(params.DEVICE)
        elif typeModel == ModelEnum.DeepVONet_FSM:
            model = DeepVONet_FSM().to(params.DEVICE)

        elif typeModel == ModelEnum.SmallDeepVONet:
            model = SmallDeepVONet().to(params.DEVICE)

        elif typeModel == ModelEnum.QuaternionDeepVONet:
            model = QuaternionDeepVONet().to(params.DEVICE)
        elif typeModel == ModelEnum.QuaternionDeepVONet_FSM:
            model = QuaternionDeepVONet_FSM().to(params.DEVICE)
        else:
            raise ValueError

        return model


