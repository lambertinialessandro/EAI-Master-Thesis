
import torch
from torch import nn


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

