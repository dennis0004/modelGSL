# -*- coding: utf-8 -*-
"""


@author: cwleung
"""

import numpy as np
import time

class GameAssignment01:
    
    def genAgentsVS(self, N, m):
#        shuffledList = []
        agentsVS = -1*np.ones((N,m), dtype=np.int)
        d = m*np.ones(N, dtype=np.int)
        
        while True:
            if np.sum(d) == 0:
                break
            agentIdx = np.argmax(d)
#            agentIdx = np.argwhere(d==np.amax(d[d>0]))[0][0]
            
            agentsAvailable = list(range(N))
            agentsAvailable.remove(agentIdx)
            for agentIdx2 in agentsVS[agentIdx]:
                if agentIdx2==-1:
                    break
                agentsAvailable.remove(agentIdx2)
            prob = d[agentsAvailable]
            prob = prob / np.sum(prob)
            
            try:
                agentVS = np.random.choice(agentsAvailable, d[agentIdx], replace=False, p=prob)
            except:
                print(agentIdx)
                print(agentsAvailable)
                print(prob)
                print(agentsVS)
                
                print('restart')
                agentsVS = -1*np.ones((N,m), dtype=np.int)
                d = m*np.ones(N, dtype=np.int)
                continue
                
            
            for agentIdx2 in agentVS:
#                shuffledList.append((agentIdx, agentIdx2))
                agentsVS[agentIdx][m-d[agentIdx]] = agentIdx2
                agentsVS[agentIdx2][m-d[agentIdx2]] = agentIdx
                d[agentIdx] -= 1
                d[agentIdx2] -= 1
        return agentsVS
    
    def genShuffledList(self, N, m):
        shuffledList = []
        agentsVS = -1*np.ones((N,m), dtype=np.int)
        d = m*np.ones(N, dtype=np.int)
        
        while True:
            if np.sum(d) == 0:
                break
            agentIdx = np.argmax(d)
#            agentIdx = np.argwhere(d==np.amax(d[d>0]))[0][0]
            
            agentsAvailable = list(range(N))
            agentsAvailable.remove(agentIdx)
            for agentIdx2 in agentsVS[agentIdx]:
                if agentIdx2==-1:
                    break
                agentsAvailable.remove(agentIdx2)
            prob = d[agentsAvailable]
            prob = prob / np.sum(prob)
            
            try:
                agentVS = np.random.choice(agentsAvailable, d[agentIdx], replace=False, p=prob)
            except:
                print(agentIdx)
                print(agentsAvailable)
                print(prob)
                print(agentsVS)
                
                print('restart')
                agentsVS = -1*np.ones((N,m), dtype=np.int)
                d = m*np.ones(N, dtype=np.int)
                continue
                
            
            for agentIdx2 in agentVS:
                shuffledList.append((agentIdx, agentIdx2))
                agentsVS[agentIdx][m-d[agentIdx]] = agentIdx2
                agentsVS[agentIdx2][m-d[agentIdx2]] = agentIdx
                d[agentIdx] -= 1
                d[agentIdx2] -= 1
        return shuffledList


if __name__ == '__main__':
    T = 1
    N = 1000
    m = 999
    
    ga = GameAssignment01()
    t1 = time.time()
    for t in range(T):
#        agentsVS = ga.genAgentsVS(N, m)
        shuffledList = ga.genShuffledList(N, m)
    t2 = time.time()
    print('time:', t2-t1)
#    print(agentsVS)
    
    Adj = np.zeros((N,N), dtype=np.int)
    for i, agentVS in enumerate(agentsVS):
        for j in agentVS:
            Adj[i,j] = 1
            Adj[j,i] = 1
#    print(Adj)
#    print(np.sum(Adj, axis=0))
#    print(np.sum(Adj, axis=1))
    
#    for agentIdx in range(N):
#        print('agent %d: '%(agentIdx) )
#        print(agentsVS[agentIdx])


