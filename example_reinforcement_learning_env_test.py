"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""

import numpy as np
import time
import Solvers
import Envs2
import datetime
import logging

# seed = 114514
seed = 324658

DATE = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
LOG_FORMAT = "%(message)s"
log_save = "./logs/" + DATE + ".log"
logging.basicConfig(filename=log_save, level=logging.DEBUG, format=LOG_FORMAT)

ROBOT = 'Panda'
EPISODES = 5000
TIMES = 1 # 1次测试
EPISODE_LENGTH = 1000
np.random.seed(seed)
load = 20

env = Envs2.ReacherEnv(ROBOT, if_headless=True, times=TIMES)
agent_teach = Solvers.Pseudo_inverse_Jacobian_solver_with_disturb(env)
agent = Solvers.RL_solver(env, train=False, seed=seed, load_path='./save/',load=load)
frams = 0

POSES = np.load("test_simple.npy")  # 500个目标点
enable_list = []  # 能否到达

logging.debug('ROBOT:,{}, Load:{}'.format(ROBOT, load))
count_not_success = 0

for episode in range(len(POSES)):

    print('Starting episode: %d,' % episode)
    state = env.reset([POSES[episode]])
    # print(state)
    agent_teach.reset()
    t0 = time.time()
    loss_c = 0
    loss_a = 0
    reward = 0
    result = 0.5
    # trace_length = round(len(agent_teach.trace) * 1.0)
    for i in range(EPISODE_LENGTH):

        disturb = agent.act(state)

        agent_teach.disturb = disturb

        action = agent_teach.act(state)
        reward, next_state, done, info = env.step(action)
        state = next_state
        loss_c += agent.model.loss_c
        loss_a += agent.model.loss_a

        if done or i == EPISODE_LENGTH - 1:
            frams += i
            if info != 'success':
                count_not_success += 1
            if info == 'success':
                result = 1
            elif info == 'collision':
                result = 0
            print(
                'Frams:{}|Steps:{}|episode_reward:{:.4f}|'
                'linear:{:.4f},orient:{:.4f},|Running Time:{:.4f}|loss_c,{:.4f},loss_a,{:.4f}|State:{}|faild:{}'.format(
                    frams, i, env.sum_record, env.linear_record / (i + 1e-6),
                                              env.orientation_record / (i + 1e-6), time.time() - t0,
                                              loss_c / (i + 1e-6),
                                              loss_a / (i + 1e-6), info, count_not_success)
            )
            logging.debug(
                'Ep:{}|Frams:{}|Steps:{}|episode_reward:{:.4f}|''linear:{:.4f},orient:{:.4f},|Running Time:{:.4f}|loss_c,{:.4f},loss_a,{:.4f}|State:{}|faild:{}'.format(
                    episode, frams, i, env.sum_record, env.linear_record / (i + 1e-6),
                                                       env.orientation_record / (i + 1e-6),
                                                       time.time() - t0,
                                                       loss_c / (i + 1e-6), loss_a / (i + 1e-6), info,
                    count_not_success))
        # if done:
            if (episode+1)%10==0: # 解决qp不精准问题，改成10次，其他100次重启一次就可以了
                env.restart()
            break

    res = np.concatenate([POSES[episode], [env.sum_record, result, i, env.linear_record / (i + 1e-6), env.orientation_record / (i + 1e-6)]])
    enable_list.append(res)
    print("result", res)

filename = str(str.lower(ROBOT)+"_reachable.npy")


np.save(filename, enable_list)
print('Done and save file in ' + filename)
env.shutdown()
