# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:58:31 2022

@author: lambe
"""

import torch.nn as nn
import torch.optim as optim

from DeepVONet import DeepVONet
from QuaternionDeepVONet import QuaternionDeepVONet

import params
from utility import PrintManager, bcolors
pm = PrintManager(params.FLAG_DEBUG_PRINT, params.FLAG_INFO_PRINT)

def build(typeModel="DeepVONet", typeCriterion="MSELoss", typeOptimizer="Adam"):
    if typeModel == "DeepVONet":
        model = DeepVONet(sizeHidden=params.BACH_SIZE).to(params.DEVICE)
    elif typeModel == "QuaternionDeepVONet":
        model = QuaternionDeepVONet(sizeHidden=params.BACH_SIZE).to(params.DEVICE)
    else:
        raise ValueError

    if typeCriterion == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError

    if typeOptimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif typeOptimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.5, weight_decay=0.5)
    else:
        raise ValueError

    print(bcolors.LIGHTRED+"Building model:"+bcolors.ENDC)
    print(bcolors.LIGHTYELLOW+"model: {}".format(typeModel)+bcolors.ENDC)
    print(bcolors.LIGHTYELLOW+"criterion: {}".format(typeCriterion)+bcolors.ENDC)
    print(bcolors.LIGHTYELLOW+"optimizer: {}".format(typeOptimizer)+bcolors.ENDC+"\n")
    return model, criterion, optimizer

def main():
    typeModel = "DeepVONet" # "DeepVONet", "QuaternionDeepVONet"
    typeCriterion = "MSELoss"
    typeOptimizer = "Adam" # "Adam", "SGD"

    build(typeModel=typeModel, typeCriterion=typeCriterion, typeOptimizer=typeOptimizer)


if __name__ == "__main__":
    main()