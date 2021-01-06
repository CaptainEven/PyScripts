"""
Dependencies:
torch: 0.3
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


# -------------------------
def q_select(A, k):
    '''
    利用quick sort的思想选择top K
    '''
    if len(A) < k:
        return A
    pivot = A[-1] # 选择枢纽
    right = [pivot] + [x for x in A[:-1] if x >= pivot]
    r_len = len(right)
    if r_len == k:
        return right
    if r_len > k:
        return q_select(right, k)
    else:
        left = [x for x in A[:-1] if x < pivot]
        return q_select(left, k - r_len) + right


for i in range(1, 10):
    print(q_select([11, 8, 4, 1, 5, 2, 7, 9], i), '\n')
# -------------------------


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 3000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) \
    else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    '''
    拟合一个函数: 输入状态, 输出每个动作的Q值
    '''

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 60)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(60, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


import os


class DQN(object):
    def __init__(self):
        if os.path.exists('./net.pkl'):  # 加载训练过的网络
            self.net = torch.load('./net.pkl')
            for name, params in self.net.named_parameters():
                print(name, ':', params.size())
            print('--Net loaded.')
        else:
            self.net = Net()  # 初始化一个未经训练的网络

        # for target updating
        self.learn_step_counter = 0

        # for storing memory
        self.memory_counter = 0

        # initialize memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))

        # input only one sample
        if np.random.uniform() < EPSILON:   # ε-greedy
            actions_value = self.net.forward(state)
            action = torch.max(actions_value, 1)[
                1].data.numpy()  # get the argmax index
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

        # get sample from memory: randomly selected batch learning
        sample_ids = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_ids, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(
            b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(
            b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_cur w.r.t the action in experience
        q_cur = self.net.forward(b_s).gather(1, b_a)  # shape (batch, 1)

        # detach from graph, don't backpropagate(don't compute grad)
        q_next = self.net.forward(b_s_).detach()  # Variable->Tensor
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(
                BATCH_SIZE, 1)  # 取每一行最大值(而非索引), reshape(batch, 1)
        loss = self.loss_func(q_cur, q_target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\n--Collecting experience...')
for i_episode in range(1000):
    # init
    s = env.reset()
    ep_r = 0

    # episode loop
    while True:
        env.render()

        # select action
        action = dqn.choose_action(s)

        # execute the action
        s_, r, done, info = env.step(action)

        # modify the reward: 计算动作后的新状态得到的奖励
        x, v, theta, ω = s_  # 四个状态: 位移, 速度, 角度, 角速度
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        # store states, action, and reward
        dqn.store_transition(s, action, r, s_) # 尝试将reward也加入memory
        ep_r += r

        # 达到memory capacity(有了一定规模的Q-table)才进行学习, 更新Q-table
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| reward: %.3f' % ep_r)

        if done:
            break  # 结束即跳出本episode
        s = s_

# save trained model
torch.save(dqn.net, './net.pkl')
print('--Dump the DQN-model done.')

# http://pytorch.org/docs/master/torch.html (pytorch官方文档)
# https://github.com/MorvanZhou/PyTorch-Tutorial
# https://www.zhihu.com/question/23765351 (softmax的作用)
# https://morvanzhou.github.io (python教程)
