
import torch
import torch.nn as nn

import params


class C_Block(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dropout_rate):
    super(C_Block, self).__init__()

    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                          padding=padding)
    self.relu = nn.ReLU(inplace=True)
    self.batch = nn.BatchNorm2d(out_ch)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    x = self.batch(x)
    x = self.drop(x)
    return x


class SmallDeepVONet(nn.Module):
    dimLSTM = params.DIM_LSTM
    hidden_size = params.HIDDEN_SIZE_LSTM

    def __init__(self):
        super(SmallDeepVONet, self).__init__()

        self.block1 = C_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = C_Block (64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = C_Block (128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block4 = C_Block (256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5 = C_Block (512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block6 = C_Block (512, 1024, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)

        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(input_size=self.dimLSTM, hidden_size=self.hidden_size,
                            num_layers=2, dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=self.hidden_size, out_features=6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.flatten(x)

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        x = self.linear_output(x)
        return x


