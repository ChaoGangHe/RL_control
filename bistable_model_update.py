import numpy as np
import math

def bistable(Q, Fc, base):
    # 参数定义
    mu = 18.65 # 质量比
    omega1 = 71.32 # 斜置弹簧固有频率
    omega2 = 177.80 # 弹性边界线性弹簧固有频率
    eta1 = 0.03 # 动子阻尼比
    eta2 = 0.0016 # 弹性边界阻尼比
    alpha = 1.1111 # 长度比
    d = 0.072 # 动子与连接质量块垂直高度
    g = 9.8 #重力加速度

    # 低通滤波器参数，此滤波器允许0-100Hz频带的随机噪声通过
    omega0 = 628  # 滤波器固有频率，omega0/2/pi转为Hz，这里是100Hz
    eta = 1.414  # 滤波器损失系数，eta=1.414时omega0就是滤波器带宽
    G0 = 1  # 滤波器增益

    V = 10 # 放大因子
    dt = 0.0002 # 计算步长，步长需要小一些，大了计算会发散

    A3 = - V * Q[4] - (eta1 * omega1 * Q[2] + omega1 ** 2 * Q[0] * (1 - alpha * d / math.sqrt((d - Q[1]) ** 2 + (Q[0]) ** 2)))
    A4 = - (eta2 * omega2 * Q[3] + omega2 ** 2 * Q[1] + mu * (omega1 ** 2) * (Q[1] - d) * (1 - alpha * d / math.sqrt((d - Q[1]) ** 2 + (Q[0]) ** 2)) - g)
    Q_1 = Q[0] + Q[2] * 0.5 * dt
    Q_2 = Q[1] + Q[3] * 0.5 * dt
    Q_3 = Q[2] + A3 * 0.5 * dt
    Q_4 = Q[3] + A4 * 0.5 * dt
    Q_5 = Q[4] + Q[5] * 0.5 * dt
    Q_6 = Q[5] - (eta * omega0 * Q[5] + omega0 ** 2 * Q[4]) * 0.5 * dt
    A3_ = - V * Q_5 - (eta1 * omega1 * Q_3 + omega1 ** 2 * Q_1 * (1 - alpha * d / math.sqrt((d - Q_2) ** 2 + Q_1 ** 2)))
    A4_ = -(eta2 * omega2 * Q_4 + omega2 ** 2 * Q_2 + mu * omega1 ** 2 * (Q_2 - d) * (1 - alpha * d / math.sqrt((d - Q_2) ** 2 + Q_1 ** 2)) - g)
    Q[0,] = Q[0] + Q_3 * dt
    Q[1] = Q[1] + Q_4 * dt
    Q[2] = Q[2] + A3_ * dt + Fc * dt
    Q[3] = Q[3] + A4_ * dt
    Q[4] = Q[4] + Q_6 * dt
    Q[5] = Q[5] - (eta * omega0 * Q_6 + omega0 ** 2 * Q_5) * dt + G0 * omega0 ** 2 * base
    global acc
    acc = Fc - eta1 * omega1 * Q[2] - omega1 ** 2 * Q[0] * ( 1 - alpha * d / ((d - Q[1]) ** 2 + (Q[0]) ** 2) ** 0.5)
    # print(acc)
    global basecount
    # print(basecount)
    acc_end[basecount] = acc
    BASE[basecount] = Q[4] * V
    basecount = basecount + 1
    ACC[0:flength-1] = ACC[1:flength]
    ACC[flength-1] = acc
    # return ACC



D = 0.001
dt = 0.0002
N = 10000  # 基础激励序列长度
base = math.sqrt(2 * D * dt) * np.random.randn(N+1)

flength = 300  # 参与神经网络训练的acc序列长度选择
ACC = np.zeros(flength)
Q = np.zeros((6, 1))
basecount = 0
acc_end = np.zeros(N)
BASE = np.zeros(N)

Fc = 0

for j in range(500):
    bistable(Q, Fc, base[basecount])

# print(Q)
print(ACC)
# print(acc_end)



