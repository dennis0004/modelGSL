# -*- coding: utf-8 -*-
"""


@author: cwleung
"""
# normal Q, Boltzmann, 1 state

import numpy as np
import random
import itertools

class Agent02:
    def __init__(self, env, lr=0.1, gamma=1, tau=1, name='IL-bolt'):
        self.name = name
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
#        self.t = 0
        
        self.Q = np.zeros(env.Nact, dtype=np.float)
    
    def getQsa(self, s, a):
        return self.Q[a]
    
    def getProbS(self, s):
        return self.softmax(self.tau*self.Q)
    
    def softmax(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        return softmax
    
    def getAction(self, s):
        legalActions = self.env.calLegalActions(s)
        subQs = self.Q[legalActions]
        
        probS = self.softmax(self.tau*subQs)
        action = np.random.choice(legalActions, p=probS)
        return action
    
    def getPolicy(self, s):
        return self.getAction(s)
    
    def train(self, s, a, r, s_):
#        sample = reward+self.gamma*self.computeValueFromQs(env, nextState)
        self.Q[a] = (1-self.lr)*self.Q[a] + self.lr*r
    
    def computeValueFromQs(self, env, state):
        actions = env.getLegalActions(state)
        subQs = self.Qs[state, actions]
        return np.max(subQs)
    
    




