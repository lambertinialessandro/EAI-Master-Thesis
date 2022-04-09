# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:54:35 2022

@author: lambe
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import quaternionFunctions as QF
from params import DIM_LSTM

class QuatC_Block(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dropout_rate):
    super(QuatC_Block, self).__init__()

    self.conv = QF.QuaternionConv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    self.relu = nn.ReLU(inplace=True)
    self.batch = nn.BatchNorm2d(out_ch)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    x = self.batch(x)
    x = self.drop(x)
    return x

class QuaternionDeepVONet(nn.Module):
    def __init__(self, sizeHidden=1):
        super(QuaternionDeepVONet, self).__init__()

        self.block1 = QuatC_Block(8, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), dropout_rate=0.2)
        self.block2 = QuatC_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(2, 2), dropout_rate=0.2)
        self.block3 = QuatC_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = QuatC_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block4 = QuatC_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = QuatC_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block5 = QuatC_Block (512, 512, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = QuatC_Block (512, 512, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block6 = QuatC_Block (512, 1024, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)

        # self.dimLSTMCell1 = DIM_LSTM
        # self.lstm1 = nn.LSTMCell(self.dimLSTMCell1, 1000)
        # self.lstm1_dropout = nn.Dropout(0.5)
        # self.lstm2 = nn.LSTMCell(1000, 1000)
        # self.lstm2_dropout = nn.Dropout(0.5)

        # self.fc = nn.Linear(in_features=1000, out_features=6)

        # self.reset_hidden_states(sizeHidden=sizeHidden, zero=True)

        self.flatten = nn.Flatten()

        self.dimLSTM = DIM_LSTM
        self.lstm = nn.LSTM(input_size=self.dimLSTM, hidden_size=1000, num_layers=2,
                            dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=1000, out_features=6)

    # def reset_hidden_states(self, sizeHidden=1, zero=True):
    #     if zero == True:
    #         self.hx1 = Variable(torch.zeros(sizeHidden, 1000))
    #         self.cx1 = Variable(torch.zeros(sizeHidden, 1000))
    #         self.hx2 = Variable(torch.zeros(sizeHidden, 1000))
    #         self.cx2 = Variable(torch.zeros(sizeHidden, 1000))
    #     else:
    #         self.hx1 = Variable(self.hx1.data)
    #         self.cx1 = Variable(self.cx1.data)
    #         self.hx2 = Variable(self.hx2.data)
    #         self.cx2 = Variable(self.cx2.data)

    #     if next(self.parameters()).is_cuda == True:
    #         self.hx1 = self.hx1.cuda()
    #         self.cx1 = self.cx1.cuda()
    #         self.hx2 = self.hx2.cuda()
    #         self.cx2 = self.cx2.cuda()

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

        # #print(x.size())
        # x = x.view(x.size(0), self.dimLSTMCell1)
        # #print(x.size())
        # self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))
        # x = self.lstm1_dropout(self.hx1)

        # self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))
        # x = self.lstm2_dropout(self.hx2)

        # x = self.fc(x)

        x = self.flatten(x)

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        x = self.linear_output(x)
        return x


