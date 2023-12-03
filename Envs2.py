from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
import numpy as np
from arms.panda import Panda

EPISODES = 50
EPISODE_LENGTH = 300
R_MAX = 0.6
R_MIN = 0.3  # 极坐标半径
THETA_MAX = np.pi / 2
THETA_MIN = np.pi / 2.5  # 极坐标仰角
PHI_MAX = np.pi * 2
PHI_MIN = 0  # 极坐标圆周范围
HEIGHT = 1.083


class ReacherEnv(object, ):
    '''
    从一个固定点出发，到达一个随机位置
    '''

    def __init__(self, arm=None, if_headless=False, times=1):
        self.pr = PyRep()
        # self.pr.set_simulation_timestep(0.05)
        self.arm = arm
        self.if_headless = if_headless

        if arm == 'Panda':
            SCENE_FILE = join(dirname(abspath(__file__)), './copp_envs/scene_reinforcement_learning_env.ttt')
            self.pr.launch(SCENE_FILE, headless=if_headless)
            self.pr.start()
            self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)  # 这两个用于控制机械臂仿真，让其能够按照速度运动，并且能够瞬移到某个特定的位置
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()  # 末端执行器夹具中间点
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_tip_positions = self.agent_ee_tip.get_position()  # 初始tip位置
        self.initial_tip_orientation = self.agent_ee_tip.get_orientation()  # 初始tip朝向
        self.joints = self.agent.joints
        self.joints_num = self.agent.get_joint_count()
        self.velocities_tip_before = np.concatenate(self.agent_ee_tip.get_velocity())
        self.times = times - 1  # 连续追踪次数，只追踪一次时该值为0
        self.count_time = self.times
        self.pos_list = self.create_pos_list()

        # rewards
        self.target_dis = 0
        self.reward_linear = 0
        self.reward_orientation = 0
        self.reward_velocities = 0
        self.reward_near = 0

        # para
        self.state_dim = len(self._get_state())
        self.action_dim = self.joints_num
        self.max_action = 1
        self.reward_ratio = 1

        # records
        self.linear_record = 0
        self.orientation_record = 0
        self.velocities_record = 0
        self.near_record = 0
        self.sum_record = 0

        # self.targets = []

    def _get_state(self):
        # print("joint_pos:", self.agent.get_joint_positions())

        return np.concatenate([
            np.array(self.agent.get_joint_positions()) / np.pi,
            # self.agent.get_joint_velocities(),
            np.array(self.target.get_position() - self.agent_ee_tip.get_position()) / np.array([1, 1, 0.4]),
            (np.array(self.initial_tip_orientation - self.agent_ee_tip.get_orientation())) / np.pi,
        ])

    def _get_state_discard(self):
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities(),
                               self.target.get_position(),
                               self.target.get_orientation(), ])

    def reset(self, POS=0):
        # Get a random position within a cuboid and set the target position
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)  # 如果为False则不回原位置
        if isinstance(POS, int):
            self.pos_list = self.create_pos_list()
        else:
            self.pos_list = POS
        # print("pos_list",self.pos_list)
        self.target.set_position(self.pos_list[-1])  # 用最后一个，接下来就用第self.count_times个

        self.initial_tip_orientation = self.agent_ee_tip.get_orientation()
        self.target.set_orientation(self.initial_tip_orientation)

        self.count_time = self.times

        # clear rewards
        # self.target_dis = 0
        self.reward_linear = 0
        self.reward_orientation = 0
        self.reward_velocities = 0
        self.reward_near = 0

        # clear records
        self.linear_record = 0
        self.orientation_record = 0
        self.velocities_record = 0
        self.near_record = 0
        self.sum_record = 0
        return self._get_state()

    def set_another_target_pos_deprecated(self, POS=0):
        '''
        不重置机械臂，只选择新的target位置
        '''
        if isinstance(POS, int):
            pos = self.random_pos()
            target_pos_now = self.target.get_position()
            while self.Cal_dis(pos, target_pos_now) < 0.1:  # 确保新生成的位置距离平方大于0.1
                pos = self.random_pos()
                # print("new dis",self.Cal_dis(pos, target_pos_now))
        else:
            pos = POS

        self.target.set_position(pos)

    def get_another_target_pos(self, pos_now, POS=0):
        '''
        生成一个新的pos，和pos_now保持距离
        '''
        if isinstance(POS, int):
            pos = self.random_pos()
            while self.Cal_dis(pos, pos_now) < 0.1:  # 确保新生成的位置距离平方大于0.3
                pos = self.random_pos()
                # print("new dis",self.Cal_dis(pos, target_pos_now))
        else:
            pos = POS
        return pos

    def create_pos_list(self):
        '''
        生成一组target position，数量和self.times+1相等
        '''
        pos_list = []
        pos_list.append(self.random_pos())
        for i in range(self.times):
            pos_list.append(self.get_another_target_pos(pos_list[i - 1]))
        return pos_list

    def step(self, action):
        self.velocities_tip_before = np.concatenate(self.agent_ee_tip.get_velocity())
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        done = False
        # Reward is negative distance to target
        reward = self.Cal_reward()
        info = ""
        if self.target_dis < 0.001:
            if self.count_time <= 0:
                done = True
                info = "success"
                # print("success end")
                # print("count_times",self.count_time)
            else:
                info = "reset"
                # print("reset")
                # print("count_times", self.count_time)
                # self.set_another_target_pos()
                self.count_time = self.count_time - 1
                self.target.set_position(self.pos_list[self.count_time])
            # reward += 10
            # self.sum_record += 10

            # print("success")
        if self.agent.check_arm_collision():
            done = True
            reward -= 75
            self.sum_record -= 75
            info = "collision"
            # self.count_time = self.times
            # self.reset()
            # print("faild, collision!")
        return reward, self._get_state(), done, info

    def Cal_dis(self, a, b):
        '''
        给a,b两个位置坐标，返回两者之间的距离的平方
        '''
        ax, ay, az = a
        bx, by, bz = b
        dis = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2  # 距离的平方
        return dis

    def Cal_reward(self):
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        ix, iy, iz = self.initial_tip_positions
        irx, iry, irz = self.initial_tip_orientation
        arx, ary, arz = self.agent_ee_tip.get_orientation()
        vlx, vly, vlz, vrx, vry, vrz = np.concatenate(self.agent_ee_tip.get_velocity())
        self.target_dis = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2  # 现在距离的平方

        a = np.array(self.target.get_position() - self.agent_ee_tip.get_position())
        if vlx**2+vly**2+vlz**2<1e-5:
            cos_ = 0
        else:
            b = np.array([vlx, vly, vlz])
            # a /= np.linalg.norm(a)
            # b /= np.linalg.norm(b)
            # 夹角cos值
            cos_ = np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b))+1e-6)

        self.reward_linear = cos_ - 1
        self.reward_near = -np.sqrt(self.target_dis) / 10
        self.reward_orientation = -np.sqrt(
            np.sqrt((arx - irx) ** 2 + (ary - iry) ** 2 + (arz - irz) ** 2)) / 100  # 角度奖励，保持和初始一致
        self.reward_velocities = -np.max(np.array(self.agent.get_joint_velocities())) / 100

        reward = self.reward_linear*self.reward_ratio + self.reward_orientation*(2-self.reward_ratio) + self.reward_near

        # record rewards
        self.linear_record += self.reward_linear
        self.orientation_record += self.reward_orientation
        self.sum_record += reward
        self.velocities_record += self.reward_velocities
        self.near_record += self.reward_near

        return reward

    def restart(self):
        self.shutdown()
        self.start_up(self.arm, self.if_headless, self.times+1) # 这里要+1，因为设定的时候-1了

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def start_up(self, arm, if_headless, times):
        self.pr = PyRep()
        # self.pr.set_simulation_timestep(0.05)

        if arm == 'Panda':
            SCENE_FILE = join(dirname(abspath(__file__)), './copp_envs/scene_reinforcement_learning_env.ttt')
            self.pr.launch(SCENE_FILE, headless=if_headless)
            self.pr.start()
            self.agent = Panda()
        else:
            SCENE_FILE = join(dirname(abspath(__file__)), './copp_envs/scene_reinforcement_learning_xmate3_env.ttt')
            self.pr.launch(SCENE_FILE, headless=if_headless)
            self.pr.start()
            self.agent = Xmate3()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)  # 这两个用于控制机械臂仿真，让其能够按照速度运动，并且能够瞬移到某个特定的位置
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()  # 末端执行器夹具中间点
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_tip_positions = self.agent_ee_tip.get_position()  # 初始tip位置
        self.initial_tip_orientation = self.agent_ee_tip.get_orientation()  # 初始tip朝向
        self.joints = self.agent.joints
        self.joints_num = self.agent.get_joint_count()
        self.velocities_tip_before = np.concatenate(self.agent_ee_tip.get_velocity())
        self.times = times - 1  # 连续追踪次数，只追踪一次时该值为0
        self.count_time = self.times
        self.pos_list = self.create_pos_list()

        # rewards
        self.target_dis = 0
        self.reward_linear = 0
        self.reward_orientation = 0
        self.reward_velocities = 0
        self.reward_near = 0

        # para
        self.state_dim = len(self._get_state())
        self.action_dim = self.joints_num
        self.max_action = 1

        # records
        self.linear_record = 0
        self.orientation_record = 0
        self.velocities_record = 0
        self.near_record = 0
        self.sum_record = 0

    def random_pos(self):
        '''
        极坐标算一个随机位置，最后高度加上第一轴的高度
        :return:
        '''
        flag = True
        position_min, position_max = [-0.5, -0.5, 0.1], [0.5, 0.5, 0.5]
        pos = list(np.random.uniform(position_min, position_max))
        while flag:  # or (pos[0]>-0.2 and np.random.rand()<0.8):
            position_min, position_max = [-0.5, -0.5, 0.1], [0.5, 0.5, 0.5]
            pos = list(np.random.uniform(position_min, position_max))
            if pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2 < 0.5 ** 2:
                flag = False
            else:
                flag = True
            pos[2] += HEIGHT

        return pos

    def Cal_FK_pos(self):
        # T = self.joints[-1].get_matrix(relative_to=None)
        T = self.agent_ee_tip.get_matrix(relative_to=None)
        pos = [T[0][3], T[1][3], T[2][3]]
        return pos

    def get_zo(self):
        """
        一个中间过程，取每一个关节齐次变换矩阵的第三列和第四列（Z轴转角和位置）
        计算雅可比矩阵的中间参数和state的一部分
        :return:
        """
        Z0i = []
        o0i = []
        for i in range(self.joints_num - 1):
            T = self.joints[i].get_matrix(relative_to=None)
            Z0i.append(np.array([T[0][2], T[1][2], T[2][2]]))
            o0i.append(np.array([T[0][3], T[1][3], T[2][3]]))
        T = self.agent_ee_tip.get_matrix(relative_to=None)  # 最后一个用tip位置
        Z0i.append(np.array([T[0][2], T[1][2], T[2][2]]))
        o0i.append(np.array([T[0][3], T[1][3], T[2][3]]))
        o0n = np.array([T[0][3], T[1][3], T[2][3]])
        return Z0i, o0i, o0n

    def Cal_Ja_discard(self, ):
        """
        Calculate the Jacobian Matrix
        :return:
        """
        J = np.zeros([6, self.joints_num])
        Z0i, o0i, o0n = self.get_zo()
        for i in range(self.joints_num):
            J[3][i] = Z0i[i][0]
            J[4][i] = Z0i[i][1]
            J[5][i] = Z0i[i][2]
            Jvi = np.cross(Z0i[i], (o0n - o0i[i]))
            J[0][i] = Jvi[0]
            J[1][i] = Jvi[1]
            J[2][i] = Jvi[2]

        return J

    def Cal_Ja(self, ):
        """
        Calculate the Jacobian Matrix
        :return:
        """
        J = np.zeros([6, self.joints_num])

        Z0i = []
        o0i = []
        for i in range(self.joints_num - 1):
            T = self.joints[i].get_matrix(relative_to=None)
            Z0i.append(np.array([T[0][2], T[1][2], T[2][2]]))
            o0i.append(np.array([T[0][3], T[1][3], T[2][3]]))
        T = self.agent_ee_tip.get_matrix(relative_to=None)  # 最后一个用tip位置
        Z0i.append(np.array([T[0][2], T[1][2], T[2][2]]))
        o0i.append(np.array([T[0][3], T[1][3], T[2][3]]))
        o0n = np.array([T[0][3], T[1][3], T[2][3]])
        for i in range(self.joints_num):
            J[3][i] = Z0i[i][0]
            J[4][i] = Z0i[i][1]
            J[5][i] = Z0i[i][2]
            Jvi = np.cross(Z0i[i], (o0n - o0i[i]))
            J[0][i] = Jvi[0]
            J[1][i] = Jvi[1]
            J[2][i] = Jvi[2]

        return J
