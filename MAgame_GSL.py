# -*- coding: utf-8 -*-
"""


@author: cwleung
"""
# generalized social learning (2p), 1 vs m

import os
import numpy as np
from collections import Counter
import pickle
import datetime
import time

import sys
sys.path.append('../MAS_Environments')
sys.path.append('../MAS_Agents')

from Environment02 import Environment02
from Agent02 import Agent02

from GameAssignment01 import GameAssignment01

simStart = 1
Nsim = 100

tStart = 0
T = 300

Nagent = 2000

Nplayer = 2

m = 1
# m = 2
# m = 5

lr = 0.1
tau = 2

Nact = 2
gameName = 'PD'
initPara = '{(0, 0)- 1}'
# initPara = '{(1, 0)- 1}'

# Nact = 3
# gameName = '3RPS'
# initPara = '{(1, 0, 0)- 1}'
# initPara = '{(0, 0, 0)- 1}'

initDist = 'pmf'

dirName = 'result_Q-boltzmann_%s_lr%.2f_tau%.1f_Q0-%s%s_N%d_m%d_aa' % (gameName, lr, tau, initDist, initPara, Nagent, m)
print(dirName)

if not os.path.exists(dirName):
    os.makedirs(dirName)

reward1s = {}
reward2s = {}

# PD
reward1s['PD'] = np.array([[2, 0], [3, 1]])
reward2s['PD'] = np.array([[2, 3], [0, 1]])

# SH
reward1s['SH'] = np.array([[3, 0], [2, 1]])
reward2s['SH'] = np.array([[3, 2], [0, 1]])

# HD
reward1s['HD'] = np.array([[-2, 2], [0, 1]])
reward2s['HD'] = np.array([[-2, 0], [2, 1]])

# 3RPS
reward1s['3RPS'] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
reward2s['3RPS'] = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

env = Environment02(Nact, reward1s[gameName], reward2s[gameName])
ga = GameAssignment01()

t1 = time.time()
for sim in range(simStart, Nsim + 1):
    xbarT = np.zeros((T + 1, Nact), dtype=np.float)
    QbarT = np.zeros((T + 1, Nact), dtype=np.float)

    if tStart == 0:
        pass
    else:
        xbarTinit = np.loadtxt(('%s/xbarT-sim%03d.txt') % (dirName, sim), delimiter=',')
        QbarTinit = np.loadtxt(('%s/QbarT-sim%03d.txt') % (dirName, sim), delimiter=',')
        xbarT[:tStart + 1] = xbarTinit[:tStart + 1]
        QbarT[:tStart + 1] = QbarTinit[:tStart + 1]

    # initialize
    if tStart == 0:
        agents = []
        for i in range(Nagent):
            agent = Agent02(env,
                            lr=lr,
                            tau=tau)
            if initPara == '{(1, 0, 0)- 1}':
                agent.Q[0] = 1
            if initPara == '{(1, 0)- 1}':
                agent.Q[0] = 1
            agents.append(agent)
    else:
        filePath1 = '%s/Agents-sim%03d_T%d.pickle' % (dirName, sim, tStart)
        f01 = open(filePath1, 'rb')
        agents = pickle.load(f01)
        f01.close()

    for t in range(tStart, T + 1):
        agentsVS = ga.genAgentsVS(Nagent, m)

        # draw actions
        actions = []
        xbar = np.zeros(Nact, dtype=np.float)
        Qbar = np.zeros(Nact, dtype=np.float)

        for agent in agents:
            action = agent.getAction(0)
            actions.append(action)
            Qbar = Qbar + agent.Q
            xbar = xbar + agent.getProbS(0)
        actions = np.array(actions)
        Qbar = Qbar / Nagent
        xbar = xbar / Nagent

        # recording
        countAction = Counter(actions)
        xbarT[t] = xbar
        QbarT[t] = Qbar

        if t % 100 == 0 and t != tStart:
            print('sim %d, time %d' % (sim, t))
            filePath1 = '%s/Agents-sim%03d_T%d.pickle' % (dirName, sim, t)
            f01 = open(filePath1, 'wb')
            pickle.dump(agents, f01)
            f01.close()
            np.savetxt(('%s/xbarT-sim%03d.txt') % (dirName, sim), xbarT, fmt='%.6f', delimiter=',')
            np.savetxt(('%s/QbarT-sim%03d.txt') % (dirName, sim), QbarT, fmt='%.6f', delimiter=',')

        if t == T:
            continue

        # play games
        for i, agent in enumerate(agents):
            subActions = actions[agentsVS[i]]
            countSubAction = Counter(subActions)

            avgReward = 0
            for a in range(Nact):
                moves = [actions[i], a]
                avgReward += countSubAction[a] * env.getRewards(moves)[0]
            avgReward /= m

            agent.train(0, actions[i], avgReward, 0)

    print('sim:', sim, 'countAction:', countAction)

t2 = time.time()
print('time:', t2 - t1)
print('done', datetime.datetime.now())




