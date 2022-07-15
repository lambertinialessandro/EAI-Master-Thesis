
import torch.nn as nn

from modules.network.models.quaternionFunctions import QuaternionConv
from modules.network.models.FSMModule import FSM


class QuatDSC_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dropout_rate=0.5):
        super(QuatDSC_Block, self).__init__()

        self.depth_conv = QuaternionConv(4, in_ch, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=in_ch//4)
        self.point_conv = QuaternionConv(in_ch, out_ch, kernel_size=1,
                                         stride=1, padding=0, groups=1)

        self.relu = nn.ReLU(inplace=True)
        self.batch = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)

        x = self.relu(x)
        x = self.batch(x)
        x = self.drop(x)
        return x


class QuaternionDSC_VONet(nn.Module):
    def __init__(self, input_size_LSTM, hidden_size_LSTM):
        super(QuaternionDSC_VONet, self).__init__()

        self.block1 = QuatDSC_Block(8, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), dropout_rate=0.2)
        self.block2 = QuatDSC_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(2, 2), dropout_rate=0.2)
        self.block3 = QuatDSC_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = QuatDSC_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block4 = QuatDSC_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = QuatDSC_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block5 = QuatDSC_Block(512, 512, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = QuatDSC_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), dropout_rate=0.2)
        self.block6 = QuatDSC_Block(512, 1024, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1), dropout_rate=0.2)

        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(input_size=input_size_LSTM, hidden_size=hidden_size_LSTM,
                            num_layers=2, dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=hidden_size_LSTM, out_features=6)

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

if __name__ == "__main__":
    import torch

    in_ch = 64
    out_ch = in_ch*4

    x = torch.rand(1, in_ch, 50, 50)

    #DC_B = DSC_Block(in_ch, out_ch, kernel_size=3, stride=1, padding=0, dropout_rate=0.5)
    #params_c = sum(p.numel() for p in C_B.parameters() if p.requires_grad)
    #out_c = C_B(x)

    #DSC_B = QuatDSC_Block(in_ch, out_ch, kernel_size=3, stride=1, padding=0, dropout_rate=0.5)
    #params_dsc = sum(p.numel() for p in DSC_B.parameters() if p.requires_grad)
    #out_dsc = DSC_B(x)

    #print(f"The standard convolution uses {params_c} parameters.")
    #print(f"The depthwise separable convolution uses {params_dsc} parameters.")

    #print(out_c.shape)
    #print(out_dsc.shape)


    depth_conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                stride=1, padding=0, groups=in_ch)
    point_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                     stride=1, padding=0, groups=1)
    params_depth_conv1 = sum(p.numel() for p in depth_conv1.parameters() if p.requires_grad)
    out_depth_conv1 = depth_conv1(x)
    params_point_conv1 = sum(p.numel() for p in point_conv1.parameters() if p.requires_grad)
    out_point_conv1 = point_conv1(out_depth_conv1)

    print(f"The standard convolution uses {params_depth_conv1} parameters.")
    print(f"The depthwise separable convolution uses {params_point_conv1} parameters.")

    print(out_depth_conv1.shape)
    print(out_point_conv1.shape)


    depth_conv2 = QuaternionConv(4, in_ch, kernel_size=3,
                                     stride=1, padding=0, groups=in_ch//4)
    point_conv2 = QuaternionConv(in_ch, out_ch, kernel_size=1,
                                     stride=1, padding=0, groups=1)
    params_depth_conv2 = sum(p.numel() for p in depth_conv2.parameters() if p.requires_grad)
    out_depth_conv2 = depth_conv2(x)
    params_point_conv2 = sum(p.numel() for p in point_conv2.parameters() if p.requires_grad)
    out_point_conv2 = point_conv2(out_depth_conv2)

    print(f"The standard convolution uses {params_depth_conv2} parameters.")
    print(f"The depthwise separable convolution uses {params_point_conv2} parameters.")

    print(out_depth_conv2.shape)
    print(out_point_conv2.shape)
