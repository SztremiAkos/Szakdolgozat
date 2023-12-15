from environments import NormalizeData
import gym
from gym.spaces import Box
import numpy as np
import random as rnd
from scipy.integrate import solve_ivp
from environments import MassSpringDamper, tEnd, tStart, MultipleDataEnv, MultipleDataY1Env, x0, t, NormalizeData, dataLength
import math


class InvertedPendulumEnvY0(gym.Env):
    def __init__(self, x,y,z):
        super(InvertedPendulumEnvY0, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.data = []
        self.current_step = 0
        self.max_steps = dataLength - 1


        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.action_space = Box(low=-1, high=1,dtype=np.float32)


    def reset(self):
        while self.x == 0 or self.y == 0 or self.z == 0:
          self.x = x = 9.81 #round(rnd.random(),2)
          self.y = y = round(rnd.random()*10,2)
          self.z = z = round(rnd.random()*10,2)
        modelParams = [self.x,self.y,self.z]
        self.data = NormalizeData(solve_ivp(InvertedPendulum, [tStart, tEnd], x0, t_eval=t, args=(modelParams)).y[0])

        self.current_step = 0
        observation = np.array([self.data[self.current_step]])
        return (observation,{})

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        observation = np.array([self.data[self.current_step]])

        reward = self.calculate_reward(action)
        if reward > 200:
          reward = 200

        return observation, reward, False ,done, {}

    def calculate_reward(self, action):
        difference = abs(self.data[self.current_step] - action)
        return (abs(1/difference) - 5)[0]


class InvertedPendulumEnvY1(gym.Env):
    def __init__(self, x,y,z):
        super(InvertedPendulumEnvY1, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.data = []
        self.current_step = 0
        self.max_steps = dataLength - 1


        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.action_space = Box(low=-1, high=1,dtype=np.float32)


    def reset(self):
        while self.x == 0 or self.y == 0 or self.z == 0:
          self.x = x = 9.81 #round(rnd.random(),2)
          self.y = y = round(rnd.random()*10,2)
          self.z = z = round(rnd.random()*10,2)
        modelParams = [self.x,self.y,self.z]
        self.data = NormalizeData(solve_ivp(InvertedPendulum, [tStart, tEnd], x0, t_eval=t, args=(modelParams)).y[1])

        self.current_step = 0
        observation = np.array([self.data[self.current_step]])
        return (observation,{})

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        observation = np.array([self.data[self.current_step]])

        reward = self.calculate_reward(action)
        if reward > 200:
          reward = 200

        return observation, reward, False ,done, {}

    def calculate_reward(self, action):
        difference = abs(self.data[self.current_step] - action)
        return (abs(1/difference) - 5)[0]
    
def InvertedPendulum(t, x, a, b, c):
    g = a
    l = b
    b = c

    dxdt = np.zeros(2)

    dxdt[0] = x[1]
    dxdt[1] = (g / l) * math.sin(x[0]) - b * x[1]

    return dxdt