
from enum import Enum

import torch.nn as nn


class CriterionEnum(Enum):
    MSELoss = "MSELoss"

class CriterionFactory():
    def __init__(self):
        pass

    @staticmethod
    def build(typeCriterion: CriterionEnum):
        if typeCriterion == CriterionEnum.MSELoss:
            criterion = nn.MSELoss()
        else:
            raise ValueError

        return criterion


