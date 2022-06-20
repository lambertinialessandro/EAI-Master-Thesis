
import torch.optim as optim

from enum import Enum


class OptimizerEnum(Enum):
    Adam = "Adam"
    SGD = "SGD"

class OptimizerFactory():
    def __init__(self):
        pass

    @staticmethod
    def build(typeOptimizer: OptimizerEnum, model):
        if typeOptimizer == OptimizerEnum.Adam:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        elif typeOptimizer == OptimizerEnum.SGD:
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=0.5)
        else:
            raise ValueError

        return optimizer


