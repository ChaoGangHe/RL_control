# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:41:53 2021

@author: admin

Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !

"""
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# %% 超参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
parser.add_argument('--load_nn', default=False, type=bool)  # 是否导入已有网络

parser.add_argument('--tau', default=0.002, type=float)  # target smoothing coefficient

parser.add_argument('--lr_A', default=1e-5, type=float)  # A网络学习率1e-5
parser.add_argument('--lr_C', default=1e-4, type=float)  # C网络学习率1e-4

parser.add_argument('--gamma', default=1, type=int)  # discounted factor
parser.add_argument('--capacity', default=2000, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=128, type=int)  # mini batch size

parser.add_argument('--episode_length', default=300, type=int)  # 回合长度
parser.add_argument('--save_interval', default=1, type=int)  # 相隔n回合存储一次网络参数
parser.add_argument('--max_episode', default=2001, type=int)  # 回合数
parser.add_argument('--update_iteration', default=300, type=int)  # 每回合更新网络参数的次数 300

parser.add_argument('--exploration_noise', default=0.2, type=float)  # 探索噪声初值

parser.add_argument('--state_scale_factor', default=10, type=float)  # 状态放大系数（放大之后输入网络） 10
parser.add_argument('--action_scale_factor', default=2, type=float)  # 动作放大系数（2网络输出后放大） 2

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cpu')

# %%路径
path = os.getcwd()

path1 = path + '\\pic2\\'
if os.path.exists('./pic2') == False:
    os.mkdir(path1)

directory = path + '\\nn2\\'
if not os.path.exists(directory)==True:
    os.mkdir(directory)

# %%读取传递通道模型参数
df = pd.read_excel('Wts0415.xlsx', sheet_name='Sheet1', header=None)
# df = pd.read_excel('Wts0126.xlsx',sheet_name='Sheet1', header=None)
data = df.values

W01 = data[:, 0]
W02 = data[:, 1]
W03 = data[:, 2]
W04 = data[:, 3]

W11 = data[:, 4]
W12 = data[:, 5]
W13 = data[:, 6]
W14 = data[:, 7]

W21 = data[:, 8]
W22 = data[:, 9]
W23 = data[:, 10]
W24 = data[:, 11]

W31 = data[:, 12]
W32 = data[:, 13]
W33 = data[:, 14]
W34 = data[:, 15]

W41 = data[:, 16]
W42 = data[:, 17]
W43 = data[:, 18]
W44 = data[:, 19]

W01r = W01[::-1]
W02r = W02[::-1]
W03r = W03[::-1]
W04r = W04[::-1]

W11r = W11[::-1]
W12r = W12[::-1]
W13r = W13[::-1]
W14r = W14[::-1]

W21r = W21[::-1]
W22r = W22[::-1]
W23r = W23[::-1]
W24r = W24[::-1]

W31r = W31[::-1]
W32r = W32[::-1]
W33r = W33[::-1]
W34r = W34[::-1]

W41r = W41[::-1]
W42r = W42[::-1]
W43r = W43[::-1]
W44r = W44[::-1]

df1 = pd.read_excel('motor0on0415.xlsx', sheet_name='data', header=None)
# df1 = pd.read_excel('motor0on.xlsx',sheet_name='Data', header=None)
ii = 20000
x0_array = df1.values[1 + ii:20001 + ii, 0]


# %% 定义环境
class Env():
    def __init__(self):
        # 振源、电机的输入均是长度300的向量（）
        self.X0 = 300 * [0]
        self.X1 = 300 * [0]
        self.X2 = 300 * [0]
        self.X3 = 300 * [0]
        self.X4 = 300 * [0]
        self.tau = 0.001
        self.max_size = 100
        self.ptr = 0
        self.state_memory = []

    def push(self, data):
        if len(self.state_memory) == self.max_size:
            self.state_memory[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.state_memory.append(data)

    def step(self, x0, x1, x2, x3, x4):
        # x0是振源, x1,x2,x3,x4电机当前输入值、也是Action
        # 更新输入向量
        self.X0.append(x0)
        self.X0.pop(0)

        self.X1.append(x1)
        self.X1.pop(0)

        self.X2.append(x2)
        self.X2.pop(0)

        self.X3.append(x3)
        self.X3.pop(0)

        self.X4.append(x4)
        self.X4.pop(0)

        # 更新传感器的输出
        self.y1 = np.array(self.X0).dot(W01r) + \
                  np.array(self.X1).dot(W11r) + \
                  np.array(self.X2).dot(W21r) + \
                  np.array(self.X3).dot(W31r) + \
                  np.array(self.X4).dot(W41r)

        self.y2 = np.array(self.X0).dot(W02r) + \
                  np.array(self.X1).dot(W12r) + \
                  np.array(self.X2).dot(W22r) + \
                  np.array(self.X3).dot(W32r) + \
                  np.array(self.X4).dot(W42r)

        self.y3 = np.array(self.X0).dot(W03r) + \
                  np.array(self.X1).dot(W13r) + \
                  np.array(self.X2).dot(W23r) + \
                  np.array(self.X3).dot(W33r) + \
                  np.array(self.X4).dot(W43r)

        self.y4 = np.array(self.X0).dot(W04r) + \
                  np.array(self.X1).dot(W14r) + \
                  np.array(self.X2).dot(W24r) + \
                  np.array(self.X3).dot(W34r) + \
                  np.array(self.X4).dot(W44r)

        self.y1 = self.y1 * 0.1
        self.y2 = self.y2 * 0.1
        self.y3 = self.y3 * 0.1
        self.y4 = self.y4 * 0.1

        self.state = np.array([self.y1, self.y2, self.y3, self.y4])

        self.push(self.state)

        def get_reward_long(state_memory):
            states = np.array(state_memory)
            S1 = states[:, 0]
            S1c = fft(S1, 200)
            P1c = abs(S1c / 200).mean()

            S2 = states[:, 1]
            S2c = fft(S2, 200)
            P2c = abs(S2c / 200).mean()

            S3 = states[:, 2]
            S3c = fft(S3, 200)
            P3c = abs(S3c / 200).mean()

            S4 = states[:, 3]
            S4c = fft(S4, 200)
            P4c = abs(S4c / 200).mean()
            return -P1c - P2c - P3c - P4c

        def get_reward_long_RMS(state_memory):
            states = np.array(state_memory)
            S1 = states[:, 0]
            S1 = np.power(S1, 2).sum() / S1.shape[0]
            P1c = np.sqrt(S1)

            S2 = states[:, 1]
            S2 = np.power(S2, 2).sum() / S2.shape[0]
            P2c = np.sqrt(S2)

            S3 = states[:, 2]
            S3 = np.power(S3, 2).sum() / S3.shape[0]
            P3c = np.sqrt(S3)

            S4 = states[:, 3]
            S4 = np.power(S4, 2).sum() / S4.shape[0]
            P4c = np.sqrt(S4)
            return -P1c - P2c - P3c - P4c

        def get_reward_long_var(state_memory):
            states = np.array(state_memory)
            S1 = states[:, 0]
            mean1 = S1.mean()
            var1 = np.power(S1 - mean1, 2).sum() / S1.shape[0]
            S2 = states[:, 1]
            mean2 = S2.mean()
            var2 = np.power(S2 - mean2, 2).sum() / S2.shape[0]
            S3 = states[:, 2]
            mean3 = S3.mean()
            var3 = np.power(S3 - mean3, 2).sum() / S3.shape[0]
            S4 = states[:, 3]
            mean4 = S4.mean()
            var4 = np.power(S4 - mean4, 2).sum() / S4.shape[0]
            return -var1 - var2 - var3 - var4

        # long_reward = get_reward_long(self.state_memory)
        long_reward = get_reward_long_RMS(self.state_memory)
        long_reward_v = get_reward_long_var(self.state_memory)
        self.done = 0

        # 定义reward
        self.reward = -(self.y1) ** 2 - (self.y2) ** 2 - (self.y3) ** 2 - (self.y4) ** 2
        # self.reward = long_reward

        return self.state, self.reward, self.done

    def reset(self):
        self.state_memory = []
        self.X0 = 300 * [0]
        self.X1 = 300 * [0]
        self.X2 = 300 * [0]
        self.X3 = 300 * [0]
        self.X4 = 300 * [0]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state


# %% 定义智能体
class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 300)
        self.l2 = nn.Linear(300, 200)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        x = self.max_action * torch.tanh(self.l5(x))
        return x


class Actor1(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor1, self).__init__()
        self.l1 = nn.Linear(int(state_dim / 4), 50)
        self.l2 = nn.Linear(50, 20)
        self.l3 = nn.Linear(80, 40)
        self.l4 = nn.Linear(40, 20)
        self.l5 = nn.Linear(20, action_dim)
        self.max_action = max_action

    def forward(self, x):
        in1, in2, in3, in4 = x.chunk(4, 1)
        x1 = torch.tanh(self.l1(in1))
        x2 = torch.tanh(self.l1(in2))
        x3 = torch.tanh(self.l1(in3))
        x4 = torch.tanh(self.l1(in4))
        x1 = torch.tanh(self.l2(x1))
        x2 = torch.tanh(self.l2(x2))
        x3 = torch.tanh(self.l2(x3))
        x4 = torch.tanh(self.l2(x4))
        xf = torch.tanh(self.l3(torch.cat((x1, x2, x3, x4), 1)))
        xf = torch.tanh(self.l4(xf))
        xf = self.max_action * torch.tanh(self.l5(xf))
        return xf


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 200)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        return x


class Critic1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic1, self).__init__()
        self.ls1 = nn.Linear(int(state_dim / 4), 100)
        self.ls2 = nn.Linear(100, 40)
        self.la1 = nn.Linear(action_dim, 40)
        self.la2 = nn.Linear(40, 40)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 20)
        self.l5 = nn.Linear(20, 1)

    def forward(self, x, u):
        in1, in2, in3, in4 = x.chunk(4, 1)
        x1 = torch.relu(self.ls1(in1))
        x2 = torch.relu(self.ls1(in2))
        x3 = torch.relu(self.ls1(in3))
        x4 = torch.relu(self.ls1(in4))
        x1 = torch.relu(self.ls2(x1))
        x2 = torch.relu(self.ls2(x2))
        x3 = torch.relu(self.ls2(x3))
        x4 = torch.relu(self.ls2(x4))
        u1 = torch.relu(self.la1(u))
        u1 = torch.relu(self.la2(u1))
        c = torch.relu(self.l3(torch.cat([x1, x2, x3, x4, u1], 1)))
        c = torch.relu(self.l4(c))
        # c = torch.relu(self.l5(c))
        c = self.l5(c)
        return c


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor1(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor1(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_A)

        self.critic = Critic1(state_dim, action_dim).to(device)
        self.critic_target = Critic1(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_C)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.loss_c_list = []
        self.loss_a_list = []
        self.t_q_list = []
        self.c_q_list = []
        self.episode = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            next_action = self.actor(next_state)
            reg_action = torch.pow(action - next_action, 2).mean() * 1000
            # print(reg_action.cpu().data.numpy())

            # target_Q = reward + (done * args.gamma * target_Q).detach()
            target_Q = reward + 1 * args.gamma * target_Q.detach()
            tq_save = target_Q.cpu().data.numpy().mean()

            # Get current Q estimate
            current_Q = self.critic(state, action)
            cq_save = current_Q.cpu().data.numpy().mean()

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            # actor_loss = -self.critic(state, self.actor(state)).mean()-reg_action
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.loss_c_list.append(critic_loss)
            self.loss_a_list.append(actor_loss)
            self.t_q_list.append(tq_save)
            self.c_q_list.append(cq_save)

            # Update the frozen target models
            if (it % 1 == 0):
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if (it % 1 == 0):
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

        self.episode += 1

        '''
        plt.figure(figsize=(16, 12))
        plt.plot(range(len(self.Q_iter)), self.Q_iter, linewidth=3, color='g')
        plt.ylabel('Q value', size=10)
        plt.xlabel('Episode No.', size=10)
        plt.savefig(path + '\\Q_value\\' + str(self.episode) + '.png')
        '''

        plt.subplot(311)
        plt.plot(range(len(self.loss_c_list)), self.loss_c_list, linewidth=3, color='g')
        plt.ylabel('loss_critic', size=10)
        plt.xlabel('Episode No.', size=10)

        plt.subplot(312)
        plt.plot(range(len(self.loss_a_list)), self.loss_a_list, linewidth=3, color='b')
        plt.ylabel('loss_actor', size=10)
        plt.xlabel('Episode No.', size=10)

        plt.subplot(313)
        plt.plot(range(len(self.t_q_list)), self.t_q_list, linewidth=3, color='g')
        plt.plot(range(len(self.c_q_list)), self.c_q_list, linewidth=3, color='b')
        plt.ylabel('Q value', size=10)
        plt.xlabel('Episode No.', size=10)

        plt.savefig(os.getcwd() + '\\Loss1\\' + str(self.episode) + '.png')

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pt')
        torch.save(self.critic.state_dict(), directory + 'critic.pt')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pt'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pt'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


# %% 测试环境
agent = DDPG(state_dim=400, action_dim=4, max_action=1)

'''
for actor_params in agent.actor.parameters():
    torch.nn.init.normal_(actor_params ,mean=0, std=0.1)
    #torch.nn.init.uniform_(actor_params,-1,1)
print("============ agent actor initialled ================")

for critic_params in agent.critic.parameters():
    torch.nn.init.normal_(critic_params, mean=0, std=0.1)
    #torch.nn.init.uniform_(critic_params,-1,1)
print("============ agent critic initialled ================")
'''

env = Env()

# 测试env
j = 0
dt = env.tau

f1 = 35
A1 = 2
fai1 = np.pi * 1 / 2

f2 = 40
A2 = 1
fai2 = np.pi * 4 / 3

f3 = 45
A3 = 2
fai3 = np.pi * 5 / 3

acc1__ = []
acc2__ = []
acc3__ = []
acc4__ = []

while j <= args.episode_length - 1:
    # 开启振源, 实际中的振源不可知
    # x0 = A1 * np.sin(2*np.pi*f1 * dt * j)
    x0 = x0_array[j]

    # x0 = A1*np.sin(2*np.pi*f1*j*dt + fai1) + A2*np.sin(2*np.pi*f2*j*dt + + fai2) + A3*np.sin(2*np.pi*f3*j*dt + fai3)

    # 电机的动作，神经网络的输出，这里关闭电机
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    # 状态，实际从传感器读取
    env.state, reward, done = env.step(x0, x1, x2, x3, x4)

    acc1__.append(env.state[0])
    acc2__.append(env.state[1])
    acc3__.append(env.state[2])
    acc4__.append(env.state[3])

    j += 1

Fs = int(1 / env.tau)
nfft = 2 * Fs
fre = Fs / nfft * (np.array(range(0, int(nfft / 2))))

Y1 = fft(acc1__, nfft)
P1 = abs(Y1 / nfft)

Y2 = fft(acc2__, nfft)
P2 = abs(Y2 / nfft)

Y3 = fft(acc3__, nfft)
P3 = abs(Y3 / nfft)

Y4 = fft(acc4__, nfft)
P4 = abs(Y4 / nfft)

plt.figure(figsize=(10, 6))
plt.subplot(221)
plt.plot(fre, P1[0:int(nfft / 2), ], '-b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Acc1 FFT')

plt.subplot(222)
plt.plot(fre, P2[0:int(nfft / 2), ], '-b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Acc2 FFT')

plt.subplot(223)
plt.plot(fre, P3[0:int(nfft / 2), ], '-b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Acc3 FFT')

plt.subplot(224)
plt.plot(fre, P4[0:int(nfft / 2), ], '-b')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Acc4 FFT')

plt.savefig(path + '\\pic\\' + 'org.png')
# plt.show()

# %%训练智能体
if args.mode == 'train':
    print("============ train agent ==============")
    if args.load_nn == True:
        agent.load()

    ep_reward__ = []
    plt.figure(figsize=(16, 12))
    for i_episode in range(args.max_episode):
        # 进入新的episode之前, 清空列表
        acc1c__ = []
        acc2c__ = []
        acc3c__ = []
        acc4c__ = []

        action1__ = []
        action2__ = []
        action3__ = []
        action4__ = []

        ep_reward = 0
        num = 0

        # 重置
        env.reset()

        s1_history = 100 * [0.0]
        s2_history = 100 * [0.0]
        s3_history = 100 * [0.0]
        s4_history = 100 * [0.0]
        s_history = s1_history + s2_history + s3_history + s4_history

        # 获取第一个s
        state = args.state_scale_factor * np.array(s_history)

        # if i_episode%40 == 0:
        #    exp_noise = args.exploration_noise*(1-i_episode/args.max_episode)
        # else:
        #    exp_noise *= 0.9 #每回合噪声衰减

        args.exploration_noise *= 0.999  # 每回合噪声衰减

        zero_pit_fill = 0  # all episode state should be filled with real valse, not zeros
        zero_pit_fill = args.episode_length - 1
        while (1):
            action = agent.select_action(state)
            # action[2] = 0
            # action[3] = 0
            action = (action + np.random.normal(0, args.exploration_noise, size=4)).clip(-1, 1)

            # x0 = A1*np.sin(2*np.pi*f1*num*dt)

            # r_offset = int((np.random.randn() + 1)*3000)
            # x0 = x0_array[r_offset+zero_pit_fill+num] #random offset

            x0 = x0_array[num]

            env.state, reward, done = env.step(x0, args.action_scale_factor * action[0],
                                               args.action_scale_factor * action[1],
                                               args.action_scale_factor * action[2],
                                               args.action_scale_factor * action[3])

            s1_history = [i * 1.0 for i in s1_history]
            s1_history.append(env.state[0])
            s1_history.pop(0)
            s2_history = [i * 1.0 for i in s2_history]
            s2_history.append(env.state[1])
            s2_history.pop(0)
            s3_history = [i * 1.0 for i in s3_history]
            s3_history.append(env.state[2])
            s3_history.pop(0)
            s4_history = [i * 1.0 for i in s4_history]
            s4_history.append(env.state[3])
            s4_history.pop(0)

            s_history = s1_history + s2_history + s3_history + s4_history

            next_state = args.state_scale_factor * np.array(s_history)

            if zero_pit_fill == args.episode_length - 1:  # no zeros, begin to feed to buffer
                agent.replay_buffer.push((state, next_state, action, reward, done))

            state = next_state

            if zero_pit_fill < args.episode_length - 1:
                zero_pit_fill += 1
                continue

            # 存入列表
            acc1c__.append(env.state[0])
            acc2c__.append(env.state[1])
            acc3c__.append(env.state[2])
            acc4c__.append(env.state[3])

            action1__.append(action[0])
            action2__.append(action[1])
            action3__.append(action[2])
            action4__.append(action[3])

            num += 1
            ep_reward += reward

            if num >= args.episode_length:
                ep_reward__.append(ep_reward)
                print("Episode:{} Total Reward: {:.1f} Explore:{:.2f}".format(i_episode, ep_reward,
                                                                              args.exploration_noise))
                if i_episode % 1 == 0:
                    Y1c = fft(acc1c__, nfft)
                    P1c = abs(Y1c / nfft)

                    Y2c = fft(acc2c__, nfft)
                    P2c = abs(Y2c / nfft)

                    Y3c = fft(acc3c__, nfft)
                    P3c = abs(Y3c / nfft)

                    Y4c = fft(acc4c__, nfft)
                    P4c = abs(Y4c / nfft)

                    t = env.tau * np.array(range(0, num))

                    plt.figure(figsize=(16, 12))

                    plt.subplot(441)
                    plt.plot(t, acc1__, "r--", label="acc1", linewidth=3)
                    plt.plot(t, acc1c__, "b-", label="acc1 Controlled", linewidth=3)
                    plt.xlabel("Time(s)")
                    plt.ylabel("Acc")
                    plt.legend()

                    plt.subplot(442)
                    plt.plot(t, acc2__, "r--", label="acc2", linewidth=3)
                    plt.plot(t, acc2c__, "b-", label="acc2 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(443)
                    plt.plot(t, acc3__, "r--", label="acc3", linewidth=3)
                    plt.plot(t, acc3c__, "b-", label="acc3 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(444)
                    plt.plot(t, acc4__, "r--", label="acc4", linewidth=3)
                    plt.plot(t, acc4c__, "b-", label="acc4 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(445)
                    plt.plot(fre, P1[0:int(nfft / 2), ], "r--", label="acc1", linewidth=3)
                    plt.plot(fre, P1c[0:int(nfft / 2), ], "b-", label="acc1 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(446)
                    plt.plot(fre, P2[0:int(nfft / 2), ], "r--", label="acc2", linewidth=3)
                    plt.plot(fre, P2c[0:int(nfft / 2), ], "b-", label="acc2 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(447)
                    plt.plot(fre, P3[0:int(nfft / 2), ], "r--", label="acc3", linewidth=3)
                    plt.plot(fre, P3c[0:int(nfft / 2), ], "b-", label="acc3 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(448)
                    plt.plot(fre, P4[0:int(nfft / 2), ], "r--", label="acc4", linewidth=3)
                    plt.plot(fre, P4c[0:int(nfft / 2), ], "b-", label="acc4 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(449)
                    plt.plot(t, action1__, 'k-', linewidth=3)
                    plt.ylabel("Action 1")
                    plt.xlabel("Time(s)")

                    plt.subplot(4, 4, 10)
                    plt.plot(t, action2__, 'k-', linewidth=3)
                    plt.ylabel("Action 2")
                    plt.xlabel("Time(s)")

                    plt.subplot(4, 4, 11)
                    plt.plot(t, action3__, 'k-', linewidth=3)
                    plt.ylabel("Action 3")
                    plt.xlabel("Time(s)")

                    plt.subplot(4, 4, 12)
                    plt.plot(t, action4__, 'k-', linewidth=3)
                    plt.ylabel("Action 4")
                    plt.xlabel("Time(s)")

                    plt.subplot(414)
                    plt.plot(range(len(ep_reward__)), ep_reward__, linewidth=3, color='g')
                    plt.ylabel('Return', size=10)
                    plt.xlabel('Episode No.', size=10)

                    # window路径是反斜杠，python不认，需加反斜杠进行转义
                    plt.savefig(path1 + str(i_episode) + '.png')
                    # plt.show()

                break  # 结束while循环

        # 结束while循环，还在for循环内
        if i_episode >= 1:
            agent.update()

        if i_episode % args.save_interval == 0:
            agent.save()

    # for循环结束
    reward_list = {"reward": ep_reward__, }
    frame1 = pd.DataFrame(reward_list)

    writer = pd.ExcelWriter('reward.xlsx')
    frame1.to_excel(writer, sheet_name='Sheet1', index=False)

    writer.save()

# %%测试智能体
if args.mode == 'test':
    print("============ test agent ==============")
    agent.load()

    for i_episode in range(1):

        # 进入新的episode之前, 清空列表
        acc1c__ = []
        acc2c__ = []
        acc3c__ = []
        acc4c__ = []

        action1__ = []
        action2__ = []
        action3__ = []
        action4__ = []

        ep_reward__ = []

        ep_reward = 0
        num = 0

        # 重置
        env.reset()

        s1_history = 100 * [0.0]
        s2_history = 100 * [0.0]
        s3_history = 100 * [0.0]
        s4_history = 100 * [0.0]
        s_history = s1_history + s2_history + s3_history + s4_history

        # 获取第一个s
        state = args.state_scale_factor * np.array(s_history)

        while (1):
            action = agent.select_action(state).clip(-1, 1)

            x0 = A1 * np.sin(2 * np.pi * f1 * num * dt + fai1) + A2 * np.sin(
                2 * np.pi * f2 * num * dt + + fai2) + A3 * np.sin(2 * np.pi * f3 * num * dt + fai3)

            env.state, reward, done = env.step(x0, args.action_scale_factor * action[0],
                                               args.action_scale_factor * action[1],
                                               args.action_scale_factor * action[2],
                                               args.action_scale_factor * action[3])

            s1_history = s1_history * 0.9
            s1_history.append(env.state[0])
            s1_history.pop(0)
            s2_history = s2_history * 0.9
            s2_history.append(env.state[1])
            s2_history.pop(0)
            s3_history = s3_history * 0.9
            s3_history.append(env.state[2])
            s3_history.pop(0)
            s4_history = s4_history * 0.9
            s4_history.append(env.state[3])
            s4_history.pop(0)

            s_history = s1_history + s2_history + s3_history + s4_history

            next_state = args.state_scale_factor * np.array(s_history)

            state = next_state

            # 存入列表
            acc1c__.append(env.state[0])
            acc2c__.append(env.state[1])
            acc3c__.append(env.state[2])
            acc4c__.append(env.state[3])

            action1__.append(action[0])
            action2__.append(action[1])
            action3__.append(action[2])
            action4__.append(action[3])

            num += 1
            ep_reward += reward

            if num >= args.episode_length:
                ep_reward__.append(ep_reward)

                if i_episode % 2 == 0:
                    Y1c = fft(acc1c__, nfft)
                    P1c = abs(Y1c / nfft)

                    Y2c = fft(acc2c__, nfft)
                    P2c = abs(Y2c / nfft)

                    Y3c = fft(acc3c__, nfft)
                    P3c = abs(Y3c / nfft)

                    Y4c = fft(acc4c__, nfft)
                    P4c = abs(Y4c / nfft)

                    t = env.tau * np.array(range(0, num))

                    plt.figure(figsize=(16, 12))

                    plt.subplot(341)
                    plt.plot(t, acc1__, "r--", label="acc1", linewidth=3)
                    plt.plot(t, acc1c__, "b-", label="acc1 Controlled", linewidth=3)
                    plt.xlabel("Time(s)")
                    plt.ylabel("Acc")
                    plt.legend()

                    plt.subplot(342)
                    plt.plot(t, acc2__, "r--", label="acc2", linewidth=3)
                    plt.plot(t, acc2c__, "b-", label="acc2 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(343)
                    plt.plot(t, acc3__, "r--", label="acc3", linewidth=3)
                    plt.plot(t, acc3c__, "b-", label="acc3 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(344)
                    plt.plot(t, acc4__, "r--", label="acc4", linewidth=3)
                    plt.plot(t, acc4c__, "b-", label="acc4 Controlled", linewidth=3)
                    plt.ylabel("Acc")
                    plt.xlabel("Time(s)")
                    plt.legend()

                    plt.subplot(345)
                    plt.plot(fre, P1[0:int(nfft / 2), ], "r--", label="acc1", linewidth=3)
                    plt.plot(fre, P1c[0:int(nfft / 2), ], "b-", label="acc1 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(346)
                    plt.plot(fre, P2[0:int(nfft / 2), ], "r--", label="acc2", linewidth=3)
                    plt.plot(fre, P2c[0:int(nfft / 2), ], "b-", label="acc2 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(347)
                    plt.plot(fre, P3[0:int(nfft / 2), ], "r--", label="acc3", linewidth=3)
                    plt.plot(fre, P3c[0:int(nfft / 2), ], "b-", label="acc3 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(348)
                    plt.plot(fre, P4[0:int(nfft / 2), ], "r--", label="acc4", linewidth=3)
                    plt.plot(fre, P4c[0:int(nfft / 2), ], "b-", label="acc4 Controlled", linewidth=3)
                    plt.xlabel('Frequency(Hz)')
                    plt.ylabel('FFT')
                    plt.legend()

                    plt.subplot(349)
                    plt.plot(t, action1__, 'k-', linewidth=3)
                    plt.ylabel("Action 1")
                    plt.xlabel("Time(s)")

                    plt.subplot(3, 4, 10)
                    plt.plot(t, action2__, 'k-', linewidth=3)
                    plt.ylabel("Action 2")
                    plt.xlabel("Time(s)")

                    plt.subplot(3, 4, 11)
                    plt.plot(t, action3__, 'k-', linewidth=3)
                    plt.ylabel("Action 3")
                    plt.xlabel("Time(s)")

                    plt.subplot(3, 4, 12)
                    plt.plot(t, action4__, 'k-', linewidth=3)
                    plt.ylabel("Action 4")
                    plt.xlabel("Time(s)")

                    # window路径是反斜杠，python不认，需加反斜杠进行转义
                    plt.savefig(path1 + 'test' + '35_40_45_1.png')
                    plt.show()

                break  # 结束while循环
