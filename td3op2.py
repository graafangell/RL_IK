import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import gym
import time
import random

torch.set_num_threads(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prelu_weight = random.random()


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class ReplayBuffer(object):
    '''
    修改过的ReplayBuffer，可以分离高奖励和低奖励的记录
    '''

    def __init__(self, max_size):
        self.capacity = max_size
        self.buffer = []
        self.position = 0
        self.size = 0
        self.success = 0
        self.fail = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_list(self, buffer_list, end_result=0, gamma=0.9, if_success = True):
        '''
        一个可以同时加入一组replay进入buffer的函数, 用类似蒙特卡洛的方法，最终结果的result会累加至前面的result
        '''
        # print("buffer_list", buffer_list)
        buffer_list.reverse() # 反转
        # print("reversed_list",reversed_list)
        buffer_list[0][2]=end_result
        for i in range(len(buffer_list)-1):
            buffer_list[i+1][2] += buffer_list[i][2]*gamma
            # buffer_list[i+1][2]=buffer_list[i][2]*gamma+buffer_list[i+1][2]*(1-gamma)  # 这里做一个修改
        buffer_list.reverse() # 再次反转回正
        for replay in buffer_list:
            self.add(replay[0],replay[1],replay[2],replay[3],replay[4])
            if if_success:
                self.success +=1
            else:
                self.fail +=1

    def add_her(self,buffer_list,end_result=0, gamma=0.9, if_success=True):
        '''
        一个基于her的经验回放
        '''
        length = len(buffer_list)
        start = random.randint(0,length-1)
        for i in range(start,length):
            buffer_list[i][0][7:]=[0,0,0,0,0,0]
            buffer_list[i][3][7:] = [0, 0, 0, 0, 0, 0]
            buffer_list[i][2]+=5
        if not if_success:
            buffer_list[-1][2]-=15
        for replay in buffer_list:
            self.add(replay[0],replay[1],replay[2],replay[3],replay[4])


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_O(self, batch_size):
        '''
        分离一组batch
        根据reward的平均值将奖励分为batch high和batch low
        '''
        batch = random.sample(self.buffer, batch_size)
        # batch_O = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch_O))
        # mean = np.median(reward)
        # for b in batch_O:
        #     if b[2] > mean:
        #         batch.append(b)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # reward = reward - np.mean(reward) # 针对机械臂接近过程中全为负奖励，奖励稀疏的问题

        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(done).to(device),
                )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_w=3e-3):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256,256)
        self.l4 = nn.Linear(256,256)
        # self.l5 = nn.Linear(128,128)
        self.l6 = nn.Linear(256, action_dim)
        # self.l6.weight.data.uniform_(-init_w, init_w)
        # self.l6.bias.data.uniform_(-init_w, init_w)

        self.max_action = max_action

    def forward(self, state):
        weight1 = torch.randn(1, device='cuda')
        weight2 = torch.randn(1, device='cuda')
        weight3 = torch.randn(1, device='cuda')
        weight4 = torch.randn(1, device='cuda')
        a = F.prelu(self.l1(state),weight1)
        a = F.prelu(self.l2(a),weight2)
        a = F.prelu(self.l3(a),weight3)
        a = F.prelu(self.l4(a),weight4)
        # a = F.prelu(self.l2(a),weight)
        # a = F.prelu(self.l3(a),weight)
        # a = F.leaky_relu_(self.l4(a))
        # a = F.leaky_relu_(self.l5(a))
        return self.max_action * torch.tanh(self.l6(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l11 = nn.Linear(state_dim + action_dim, 256)
        self.l12 = nn.Linear(256, 256)
        self.l13 = nn.Linear(256, 256)
        self.l14 = nn.Linear(256, 256)
        # self.l15 = nn.Linear(128,128)
        self.l16 = nn.Linear(256, 1)

        # Q2 architecture
        self.l21 = nn.Linear(state_dim + action_dim, 256)
        self.l22 = nn.Linear(256, 256)
        self.l23 = nn.Linear(256,256)
        self.l24 = nn.Linear(256,256)
        # self.l25 = nn.Linear(128,128)
        self.l26 = nn.Linear(256, 1)

        # self.l16.weight.data.uniform_(-init_w, init_w)
        # self.l26.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        weight11 = torch.randn(256, device='cuda')
        weight12 = torch.randn(256, device='cuda')
        weight13 = torch.randn(256, device='cuda')
        weight14 = torch.randn(256, device='cuda')
        q1 = F.prelu(self.l11(sa),weight11)
        q1 = F.prelu(self.l12(q1),weight12)
        q1 = F.prelu(self.l13(q1),weight13)
        q1 = F.prelu(self.l14(q1),weight14)
        # q1 = F.leaky_relu_(self.l15(q1))
        q1 = self.l16(q1)

        weight21 = torch.randn(256, device='cuda')
        weight22 = torch.randn(256, device='cuda')
        weight23 = torch.randn(256, device='cuda')
        weight24 = torch.randn(256, device='cuda')
        q2 = F.prelu(self.l21(sa),weight21)
        q2 = F.prelu(self.l22(q2),weight22)
        q2 = F.prelu(self.l23(q2),weight23)
        q2 = F.prelu(self.l24(q2),weight24)
        # q2 = F.leaky_relu_(self.l25(q2))
        q2 = self.l26(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        weight11 = torch.randn(256, device='cuda')
        weight12 = torch.randn(256, device='cuda')
        weight13 = torch.randn(256, device='cuda')
        weight14 = torch.randn(256, device='cuda')
        q1 = F.prelu(self.l11(sa),weight11)
        q1 = F.prelu(self.l12(q1),weight12)
        q1 = F.prelu(self.l13(q1),weight13)
        q1 = F.prelu(self.l14(q1),weight14)
        # q1 = F.leaky_relu_(self.l15(q1))
        q1 = self.l16(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=1e-4)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.loss_c = 0
        self.loss_a = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def save(self, filename):
        print("save file: ./save/" + filename)
        torch.save(self.critic.state_dict(), "./save/" + filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), "./save/" + filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), "./save/" + filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), "./save/" + filename + "_actor_optimizer")

    def load(self, filename):
        print("load file: ./save/" + filename)
        self.critic.load_state_dict(torch.load("./save/" + filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load("./save/" + filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load("./save/" + filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load("./save/" + filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# def main(seed):
#     # ENV_NAME = "Pendulum-v0"
#     ENV_NAME = "BipedalWalker-v2"
#
#     env = gym.make(ENV_NAME)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])
#     min_action = float(env.action_space.low[0])
#     expl_noise = 0.25
#     print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:',
#           min_action)
#
#     random_seed = seed
#     Max_episode = 5000
#
#     if random_seed:
#         print("Random Seed: {}".format(random_seed))
#         torch.manual_seed(random_seed)
#         env.seed(random_seed)
#         np.random.seed(random_seed)
#         random.seed(random_seed)
#
#     kwargs = {
#         "state_dim": state_dim,
#         "action_dim": action_dim,
#         "max_action": max_action,
#     }
#     model = TD3(**kwargs)
#     # replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
#     replay_buffer = ReplayBuffer(max_size=int(1e6))
#     frams = 0
#     soft_reward = -150
#     for episode in range(Max_episode):
#         s, done = env.reset(), False
#         episode_reward = 0
#         steps = 0
#         expl_noise *= 0.999
#         t0 = time.time()
#
#         '''Interact & trian'''
#         while not done:
#             steps += 1
#
#             a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
#                  ).clip(-max_action, max_action)
#             s_prime, r, done, info = env.step(a)
#             # env.render()
#
#             # Tricks for BipedalWalker
#             # if r <= -100:
#             if not done:
#                 # r = -1
#                 replay_buffer.add(s, a, r, s_prime, True)
#             else:
#                 replay_buffer.add(s, a, r, s_prime, False)
#
#             if replay_buffer.size > 1000: model.train(replay_buffer)
#
#             s = s_prime
#             episode_reward += r
#             frams += 1
#
#         soft_reward = soft_reward * 0.95 + episode_reward * 0.05
#         print(
#             'Episode: {} | Episode Reward: {:.4f} | Soft Reward:{:.4f} | Frams: {} | Steps: {} | Running Time: {:.4f} | loss c: {} loss a: {}'.format(
#                 episode, episode_reward, soft_reward, frams, steps, time.time() - t0, model.loss_c, model.loss_a)
#         )
#
#     env.close()

#
# if __name__ == '__main__':
#     main(seed=114514)
