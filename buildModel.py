# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:58:31 2022

@author: lambe
"""

import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML, display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torchvision import models
import torch.optim as optim

import params
from utility import PrintManager, bcolors

class C_Block(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dropout_rate):
    super(C_Block, self).__init__()

    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    self.relu = nn.ReLU(inplace=True)
    self.batch = nn.BatchNorm2d(out_ch)
    self.drop = nn.Dropout(dropout_rate)
      
  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    x = self.batch(x)
    x = self.drop(x)
    return x

class DeepVONet(nn.Module):
    def __init__(self, sizeHidden=1):
        super(DeepVONet, self).__init__()

        self.block1 = C_Block(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dropout_rate=0.2)
        self.block2 = C_Block (64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.2)
        self.block3 = C_Block (128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = C_Block (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.2)
        self.block4 = C_Block (256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = C_Block (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.2)
        self.block5 = C_Block (512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = C_Block (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.2)
        self.block6 = C_Block (512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.2)

        self.dimLSTMCell1 = 1024*5*2
        self.lstm1 = nn.LSTMCell(self.dimLSTMCell1, 1000)
        self.lstm1_dropout = nn.Dropout(0.5)
        self.lstm2 = nn.LSTMCell(1000, 1000)
        self.lstm2_dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(in_features=1000, out_features=6)

        self.reset_hidden_states(sizeHidden=sizeHidden, zero=True)

    def reset_hidden_states(self, sizeHidden=1, zero=True):
        if zero == True:
            self.hx1 = Variable(torch.zeros(sizeHidden, 1000))
            self.cx1 = Variable(torch.zeros(sizeHidden, 1000))
            self.hx2 = Variable(torch.zeros(sizeHidden, 1000))
            self.cx2 = Variable(torch.zeros(sizeHidden, 1000))
        else:
            self.hx1 = Variable(self.hx1.data)
            self.cx1 = Variable(self.cx1.data)
            self.hx2 = Variable(self.hx2.data)
            self.cx2 = Variable(self.cx2.data)

        if next(self.parameters()).is_cuda == True:
            self.hx1 = self.hx1.cuda()
            self.cx1 = self.cx1.cuda()
            self.hx2 = self.hx2.cuda()
            self.cx2 = self.cx2.cuda()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block3_1(x)
        x = self.block4(x)
        x = self.block4_1(x)
        x = self.block5(x)
        x = self.block5_1(x)
        x = self.block6(x)

        #print(x.size())
        x = x.view(x.size(0), self.dimLSTMCell1)
        #print(x.size())
        self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))
        x = self.hx1
        x = self.lstm1_dropout(x)

        self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))
        x = self.hx2
        #print(x.size())
        x = self.lstm2_dropout(x)
        
        x = self.fc(x)
        return x



def main():
    model = DeepVONet(sizeHidden=params.BACH_SIZE)
    #print(model)

    
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.5)

    return model, criterion, optimizer

if __name__ == "__main__":
    main()