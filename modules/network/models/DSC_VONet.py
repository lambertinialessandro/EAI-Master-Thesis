import torch.nn as nn


class DSC_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dropout_rate=0.5):
        super(DSC_Block, self).__init__()

        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1,
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


class DSC_VONet(nn.Module):
    def __init__(self, input_size_LSTM, hidden_size_LSTM):
        super(DSC_VONet, self).__init__()

        self.block1 = DSC_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = DSC_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = DSC_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = DSC_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block4 = DSC_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = DSC_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block5 = DSC_Block(512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = DSC_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block6 = DSC_Block(512, 1024, kernel_size=(3, 3), stride=(2, 2),
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

    x = torch.rand(5, 10, 50, 50)

    C_B = C_Block(10, 32, kernel_size=3, stride=1, padding=0, dropout_rate=0.5)
    params_c = sum(p.numel() for p in C_B.parameters() if p.requires_grad)
    out_c = C_B(x)

    DSC_B = DSC_Block(10, 32, kernel_size=3, stride=1, padding=0, dropout_rate=0.5)
    params_dsc = sum(p.numel() for p in DSC_B.parameters() if p.requires_grad)
    out_dsc = DSC_B(x)

    print(f"The standard convolution uses {params_c} parameters.")
    print(f"The depthwise separable convolution uses {params_dsc} parameters.")

    print(out_c.shape)
    print(out_dsc.shape)


