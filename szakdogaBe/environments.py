import gym
from gym.spaces import Box
import numpy as np
import random as rnd
from scipy.integrate import solve_ivp

dataLength = 2000
tStart = float(0)
tEnd = float(20)
frequency = dataLength
t = np.linspace(0, 20, frequency)
x0 = [1, 0]

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class MultipleDataEnv(gym.Env):
    def __init__(self, x,y,z):
        super(MultipleDataEnv, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.data = []  # Time series data (numpy array)
        self.current_step = 0  # Current step in the time series
        self.max_steps = dataLength - 1  # Maximum number of steps

        # Observation space: Single value representing the current data point
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Action space: Example discrete action space
        self.action_space = Box(low=-1, high=1 ,dtype=np.float32)
        #self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float64)  # Two possible actions (e.g., buy/sell)

    def reset(self):
        while self.x == 0 or self.y == 0 or self.z == 0:
          self.x = x = round(rnd.random(),2)
          self.y = y = round(rnd.random(),2)
          self.z = z = round(rnd.random(),2)
        modelParams = [self.x,self.y,self.z]
        position = solve_ivp(MassSpringDamper, [tStart, tEnd], x0, t_eval=t, args=(modelParams)).y[0]
        self.data = position
        # Reset the environment to the initial state
        self.current_step = 0
        observation = np.array([self.data[self.current_step]])
        return (observation,{})

    def step(self, action):
        # Take a step in the environment based on the chosen action
        self.current_step += 1
        done = self.current_step >= self.max_steps
        # Get the next observation
        observation = np.array([self.data[self.current_step]])
        # Calculate reward (modify this based on your specific task)
        reward = self.calculate_reward(action)
        if reward > 200:
          reward = 200

        return observation, reward, False ,done, {}

    def calculate_reward(self, action):
        # diff2 = abs(self.data[self.current_step][1] - action[1])
        difference = abs(self.data[self.current_step] - action)
        return (abs(1/difference) - 5)[0] #+ (abs(1/diff2) -5)


class MultipleDataY1Env(gym.Env):
    def __init__(self, x,y,z):
        super(MultipleDataY1Env, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        self.data = []  # Time series data (numpy array)
        self.current_step = 0  # Current step in the time series
        self.max_steps = dataLength - 1  # Maximum number of steps

        # Observation space: Single value representing the current data point
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Action space: Example discrete action space
        self.action_space = Box(low=-1, high=1 ,dtype=np.float32)
        #self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float64)  # Two possible actions (e.g., buy/sell)

    def reset(self):
        while self.x == 0 or self.y == 0 or self.z == 0:
          self.x = x = round(rnd.random(),2)
          self.y = y = round(rnd.random(),2)
          self.z = z = round(rnd.random(),2)
        modelParams = [self.x,self.y,self.z]
        self.data = solve_ivp(MassSpringDamper, [tStart, tEnd], x0, t_eval=t, args=(modelParams)).y[1]
        # Reset the environment to the initial state
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
    
def MassSpringDamper(t, y, a, b, c):
    dxdt = np.zeros(2)

    dxdt[0] = y[1]
    dxdt[1] = -(c / a) * y[0] - (b / a) * y[1]
    return dxdt