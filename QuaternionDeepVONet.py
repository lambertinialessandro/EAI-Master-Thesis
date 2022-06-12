
import torch
import torch.nn as nn

from quaternionFunctions import QuaternionConv
import params


class QuatC_Block(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dropout_rate):
    super(QuatC_Block, self).__init__()

    self.conv = QuaternionConv(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
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


class QuaternionDeepVONet(nn.Module):
    def __init__(self):
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

        self.flatten = nn.Flatten()

        self.dimLSTM = params.DIM_LSTM
        self.hidden_size = params.HIDDEN_SIZE_LSTM
        self.lstm = nn.LSTM(input_size=self.dimLSTM, hidden_size=self.hidden_size, num_layers=2,
                            dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=1000, out_features=6)

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

        x = self.flatten(x)

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        x = self.linear_output(x)
        return x


class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd
  def forward(self, x):
    return self.lambd(x)


class FSM(nn.Module):
  def __init__(self, debug=False):
    super(FSM, self).__init__()
    self.debug = debug

    self.act = nn.ReLU()

    # [64, 512, 3, 3], expected input[1, 256, 24, 16]
    # [32, 256, 3, 3], expected input[10, 1024, 5, 2]

    C = 1024 # 256
    Co = int(C // 8)

    self.b1_1 = torch.nn.Conv2d(C, Co, kernel_size=3, stride=1, padding=1)
    self.b1_2 = torch.nn.BatchNorm2d(Co)

    self.batch, self.dim1, self.dim2, channels = 10, 5, 2, 128 # 24, 16, 32 # 14, 12, 128
    Ch = (channels // 2)
    if Ch < 1:
      Ch = 1
    self.Ch = Ch

    self.b2 = torch.nn.Conv2d(Co, Ch, kernel_size=1, stride=1, padding=0)
    self.b3 = torch.nn.Conv2d(Co, Ch, kernel_size=1, stride=1, padding=0)
    self.b4 = torch.nn.Conv2d(Co, Ch, kernel_size=1, stride=1, padding=0)

    self.b6 = torch.nn.ConvTranspose2d(Ch, channels, kernel_size=1, stride=1, padding=0)

    self.b7_1 = torch.nn.ConvTranspose2d(channels, C, kernel_size=3, stride=1, padding=1)
    self.b7_2 = torch.nn.BatchNorm2d(C)


  def forward(self, x):
    if self.debug:
      print("x.shape : ", x.shape)

    b1_1 = self.act(self.b1_1(x))
    if self.debug:
      print("b1_1.shape : ", b1_1.shape)
    b1_2 = self.b1_2(b1_1)
    if self.debug:
      print("b1_2.shape : ", b1_2.shape)

    b2 = self.act(self.b2(b1_2))
    if self.debug:
      print("b2.shape : ", b2.shape)
    b2 = torch.reshape(b2, (self.Ch, -1, 1, 1))
    if self.debug:
      print("b2.shape : ", b2.shape)
    b3 = self.act(self.b3(b1_2))
    if self.debug:
      print("b3.shape : ", b3.shape)
    b3 = torch.reshape(b3, (self.Ch, -1, 1, 1))
    if self.debug:
      print("b3.shape : ", b3.shape)
    b4 = self.act(self.b4(b1_2))
    if self.debug:
      print("b4.shape : ", b4.shape)
    b4 = torch.reshape(b4, (self.Ch, -1, 1, 1))
    if self.debug:
      print("b4.shape : ", b4.shape)

    b5 = b2@b3 # dot
    if self.debug:
      print("b5.shape : ", b5.shape)
    b5_size = b5.shape
    Lambda = LambdaLayer(lambda z: (1. / float(b5_size[-1])) * z)
    b5 = Lambda(b5)
    if self.debug:
      print("b5.shape : ", b5.shape)

    b6 = b5@b4 # dot
    if self.debug:
      print("b6.shape : ", b6.shape)
    b6 = torch.reshape(b6, (self.batch, self.Ch, self.dim1, self.dim2))
    if self.debug:
      print("b6.shape : ", b6.shape)
    b6 = self.act(self.b6(b6))
    if self.debug:
      print("b6.shape : ", b6.shape)
    b6 = torch.add(b1_2, b6)
    if self.debug:
      print("b6.shape : ", b6.shape)

    b7_1 = self.act(self.b7_1(b6))
    if self.debug:
      print("b7_1.shape : ", b7_1.shape)
    b7_2 = self.b7_2(b7_1)
    if self.debug:
      print("b7_2.shape : ", b7_2.shape)
    out = torch.add(b7_2, x)
    if self.debug:
      print("out.shape : ", out.shape)

    return out


class QuaternionDeepVONet_FSM(nn.Module):
    def __init__(self):
        super(QuaternionDeepVONet_FSM, self).__init__()

        self.block1 = QuatC_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = QuatC_Block (64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = QuatC_Block (128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = QuatC_Block (256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block4 = QuatC_Block (256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = QuatC_Block (512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block5 = QuatC_Block (512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = QuatC_Block (512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block6 = QuatC_Block (512, 1024, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)

        self.fsm_block = FSM()
        self.flatten = nn.Flatten()

        self.dimLSTM = params.DIM_LSTM
        self.hidden_size = params.HIDDEN_SIZE_LSTM
        self.lstm = nn.LSTM(input_size=self.dimLSTM, hidden_size=self.hidden_size, num_layers=2,
                            dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=1000, out_features=6)

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

        x = self.fsm_block(x)
        x = self.flatten(x)
        print(x.size())

        x, _ = self.lstm(x)
        print(x.size())
        x = self.lstm_dropout(x)

        x = self.linear_output(x)
        return x

