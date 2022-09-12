
import torch.nn as nn

from modules.network.models.FSMModule import FSM


class C_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dropout_rate=0.2):
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

class DeepVONet_LSTM(nn.Module):
    def __init__(self, input_size_LSTM, hidden_size_LSTM):
        super(DeepVONet_LSTM, self).__init__()

        self.input_size_LSTM = input_size_LSTM
        self.hidden_size_LSTM = hidden_size_LSTM

        self.block1 = C_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = C_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = C_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = C_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block4 = C_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block5 = C_Block(512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block6 = C_Block(512, 1024, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)

        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(input_size=self.input_size_LSTM, hidden_size=self.hidden_size_LSTM,
                            num_layers=2, dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=self.hidden_size_LSTM, out_features=6)

    def forward(self, x, h):
        h = tuple([elem.data for elem in h])

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

        print("x: ", x.size())
        print("h: ", [h[i].size() for i in range(len(h))])
        x, h = self.lstm(x, h)

        x = self.lstm_dropout(x)
        x = self.linear_output(x)

        return x, h

    def init_hidden(self, batch_size, device):
        w = next(self.parameters()).data
        h = w.new(2, batch_size, self.hidden_size_GRU).zero_().to(device)
        return h


class DeepVONet_GRU(nn.Module):
    def __init__(self, input_size_GRU, hidden_size_GRU):
        super(DeepVONet_GRU, self).__init__()

        self.input_size_GRU = input_size_GRU
        self.hidden_size_GRU = hidden_size_GRU

        self.block1 = C_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = C_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = C_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = C_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block4 = C_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block5 = C_Block(512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block6 = C_Block(512, 1024, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)

        self.flatten = nn.Flatten()

        self.gru = nn.GRU(input_size=self.input_size_GRU, hidden_size=self.hidden_size_GRU,
                          num_layers=2, dropout=0.5, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=self.hidden_size_GRU, out_features=6)

    def forward(self, x, h):
        h = h.data
        #h = tuple([elem.data for elem in h])

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

        print("x: ", x.size())
        print("h: ", [h[i].size() for i in range(len(h))])
        x, h = self.gru(x, h)

        x = self.lstm_dropout(x)
        x = self.linear_output(x)

        return x, h

    def init_hidden(self, batch_size, device):
        w = next(self.parameters()).data
        h = w.new(batch_size, self.hidden_size_GRU).zero_().to(device)
        return h


class DeepVONet_FSM(nn.Module):
    def __init__(self, input_size_LSTM, hidden_size_LSTM):
        super(DeepVONet_FSM, self).__init__()

        self.block1 = C_Block(6, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), dropout_rate=0.2)
        self.block2 = C_Block(64, 128, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3 = C_Block(128, 256, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), dropout_rate=0.2)
        self.block3_1 = C_Block(256, 256, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block4 = C_Block(256, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block4_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block5 = C_Block(512, 512, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)
        self.block5_1 = C_Block(512, 512, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1, 1), dropout_rate=0.2)
        self.block6 = C_Block(512, 1024, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), dropout_rate=0.2)

        self.fsm_block = FSM()
        self.flatten = nn.Flatten()

        self.linear_output1 = nn.Linear(in_features=input_size_LSTM, out_features=1000)
        self.lstm_dropout1 = nn.Dropout(0.5)
        self.linear_output2 = nn.Linear(in_features=1000, out_features=100)
        self.lstm_dropout2 = nn.Dropout(0.5)

        self.linear_output = nn.Linear(in_features=100, out_features=6)

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


        x = self.linear_output1(x)
        x = self.lstm_dropout1(x)
        x = self.linear_output2(x)
        x = self.lstm_dropout2(x)

        x = self.linear_output(x)
        return x


