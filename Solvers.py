import numpy as np
import torch
import td3op

import roboticstoolbox as rtb  # for qp solver


ALPHA = 1
panda_rtb = rtb.models.Panda()  # for qp solver
# panda_rtb = 0  # 没有rtb代替一下
INTERVAL = 2

class RL_solver():
    '''

    '''

    def __init__(self, env, train=False, seed=1, load_path='./save/',load=None):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # self.max_action = env.max_action
        self.max_action = 1
        self.expl_noise = 0.25
        self.if_train = train
        self.alpha = ALPHA
        # self.alpha = 1
        print('  state_dim:', self.state_dim, '  action_dim:', self.action_dim, '  max_a:', self.max_action)

        random_seed = seed
        if random_seed:
            print("Random Seed: {}".format(random_seed))
            torch.manual_seed(random_seed)
            # env.seed(random_seed)
            np.random.seed(random_seed)
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action, }
        # self.model = pg.PG(**kwargs)
        self.model = td3op.TD3(**kwargs)

        if load:
            # self.model.load(load_path+str(load))
            self.model.load(str(load))

    def act(self, state):
        """

        :return:
        """
        a = self.model.select_action(state)
        # a = np.clip(a, -self.alpha, self.alpha)
        a = a * 1
        # self.model.select_action_test(self, state)

        return a

    def learn(self, replay_buffer):
        """

        :return:
        """
        if self.if_train:
            self.model.train(replay_buffer)

    def reset(self):
        pass

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

class Pseudo_inverse_Jacobian_solver_with_disturb(object):
    """

    """

    def __init__(self, env):
        self.env = env
        self.alpha = ALPHA
        self.trace_num = 200
        self.target_pos = env.target.get_position()
        self.target_ori = env.target.get_orientation()
        self.agent_pos = env.agent_ee_tip.get_position()
        self.agent_ori = env.agent_ee_tip.get_orientation()
        self.trace = self.generate_trace()
        self.interval = INTERVAL
        self.step = self.trace_num * self.interval
        self.damping = 0.1  # lambda for DLS
        self.joint_num = self.env.joints_num
        self.disturb = np.zeros(self.joint_num)
        # self.bias = np.array([-1.4901e-08, 6.2515e-08, 7.5526e-01])

    def reset(self):
        self.target_pos = self.env.target.get_position()
        self.target_ori = self.env.target.get_orientation()
        self.agent_pos = self.env.agent_ee_tip.get_position()
        self.agent_ori = self.env.agent_ee_tip.get_orientation()
        self.trace = self.generate_trace()
        self.step = self.trace_num * self.interval

    def generate_trace(self):
        '''
        生成位置序列
        :return:
        '''
        # begin = np.concatenate([self.agent_pos, self.agent_ori])
        # end = np.concatenate([self.target_pos, self.target_ori])
        begin = np.array(self.agent_pos)
        end = np.array(self.target_pos)
        ax, ay, az = begin
        tx, ty, tz = end
        length = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        self.trace_num = round(length / 0.005)
        trace = np.linspace(begin, end, self.trace_num + 1)
        trace = np.delete(trace, 0, axis=0)
        return trace

    def act(self, state):
        # del state
        Ja_mat = self.env.Cal_Ja()
        FK_pos = np.array(self.env.agent_ee_tip.get_position())
        delta_x = np.concatenate([np.array(self.trace[0]) - FK_pos, [0, 0, 0]])
        # ratio = np.sqrt(np.sum((FK_pos-np.array(self.target_pos))**2))/np.sqrt(np.sum((np.array(self.agent_pos)-np.array(self.target_pos))**2))
        # ratio = 0 if len(self.trace)-5<0 else len(self.trace)-5
        self.disturb = self.disturb * 1
        a = delta_x + np.matmul(Ja_mat, self.disturb)
        action = self.alpha * np.matmul(np.linalg.pinv(Ja_mat), a) - self.disturb  # 增加了扰动的右伪逆解法
        if len(self.trace) > 1 and self.step % self.interval == 0:
            self.trace = np.delete(self.trace, 0, axis=0)
        self.step -= 1
        # max = np.max(action)
        # action = action/max
        action = np.clip(action, -self.alpha, self.alpha)  # 限制最大速度为alpha
        return action

    def learn(self, replay_buffer):
        # del replay_buffer
        pass


