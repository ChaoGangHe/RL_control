import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成sin函数数据
x = np.linspace(0, 100, num=1000)
y = np.sin(x)

# 将数据转换为CNN-LSTM输入格式
window_size = 50
X = []
Y = []
for i in range(len(y) - window_size):
    X.append(y[i:i+window_size])
    Y.append(y[i+window_size])
X = torch.tensor(X).float().unsqueeze(1) # shape: (num_samples, 1, window_size)
Y = torch.tensor(Y).float().unsqueeze(1) # shape: (num_samples, 1)

# 定义CNN-LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # CNN
        x = self.conv(x)
        x = x.permute(0, 2, 1) # 交换tensor的维度
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :] # 取最后一个时间步的输出
        # 全连接层
        x = self.fc(x)
        return x

# 训练模型
model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 预测
model.eval()
with torch.no_grad():
    x_test = y[-window_size:].reshape(1, 1, -1)
    for i in range(200):
        y_pred = model(torch.tensor(x_test))
        x_test = torch.cat((x_test[:, :, 1:], y_pred.unsqueeze(1)), dim=2)

    y_pred = y_pred.squeeze().numpy()
    plt.plot(x, y, label='True Sin')
    plt.plot(np.arange(len(y), len(y)+len(y_pred)), y_pred, label='Predicted Sin')
    plt.legend()
    plt.show()

class Actor1(nn.Module):


    def __init__(self, state_dim, action_dim, max_action):
        super(Actor1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 4)
        self.max_action = max_action

    def forward(self, x):
        # CNN
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # 交换tensor的维度
        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        # 全连接层
        x = self.fc(x)
        return self.max_action * x


class Complex_CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size, stride, padding, dropout_rate):
        super(Complex_CNN_LSTM_Attention, self).__init__()

        # 定义CNN层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 定义LSTM层
        self.lstm1 = nn.LSTM(input_size=64 * 24, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout_rate)
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout_rate)

        # 定义全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

        # 定义注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # CNN层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 将CNN层的输出转换为LSTM层的输入
        x = x.view(x.size(0), -1, x.size(2))

        # LSTM层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # 注意力层
        attention_weights = self.attention(x)
        x = torch.sum(attention_weights * x, dim=1)

        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return x
class Actor1(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor1, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, padding=1),
            torch.nn.Conv1d(32, 128, 3, padding=1),
            torch.nn.MaxPool1d(3, 2, padding=1),
            Inception(192, 32, 48, 64, 8, 16, 16),
            torch.nn.MaxPool1d(3, 2, padding=1),
        )
        self.Linear_max_pool = torch.nn.MaxPool1d(5,3)
        self.Linear_action = torch.nn.Sequential(
            torch.nn.Linear(3584,1024),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(1024,512),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 128),
            torch.nn.Linear(128, action_dim)
        )
        self.max_action = max_action
class Inception(torch.nn.Module):
    def __init__(self,in_channels=56,ch1=64,ch3_reduce=96,ch3=128,ch5_reduce=16,ch5=32,pool_proj=32):
        super(Inception, self).__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,ch1,kernel_size=1), #[56,64]
            torch.nn.BatchNorm1d(ch1)
        )

        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, ch3_reduce, kernel_size=1),[56,96]
            torch.nn.BatchNorm1d(ch3_reduce),
            torch.nn.Conv1d(ch3_reduce, ch3, kernel_size=3, padding=1),96->128
            torch.nn.BatchNorm1d(ch3),
        )

        self.branch5 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, ch5_reduce, kernel_size=1),56->16
            torch.nn.BatchNorm1d(ch5_reduce),
            torch.nn.Conv1d(ch5_reduce, ch5, kernel_size=5, padding=2),16->32
            torch.nn.BatchNorm1d(ch5),
        )

        self.branch_pool = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            torch.nn.Conv1d(in_channels, pool_proj, kernel_size=1)56->32
        )

    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch3(x),self.branch5(x),self.branch_pool(x)],1)
