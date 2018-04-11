import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.optimize import fsolve

def f(x,k):
    return x**(k+1) - 2*x + 1

def count(hours):
    return np.random.normal(hours*FPH, np.sqrt(hours*FSTD**2))

DELTA = 20 #Discretize state space

DAYS = 100
TYPES = 5 #1 indexed
e = fsolve(f, .5, (TYPES))[0]
P = np.array([e**i for i in range(1,TYPES+1)])

#Action space <hours> ranges from 0 to 12 hours

#Fish per hour and STD on FPH
FPH = 20
FSTD = 5

#Pricing

#Gas pricing in dollars per gallon
GAS_MEAN = 3
GAS_PERIOD = 100 #in days
GAS_AMPLITUDE = .5
GAS_STD = .1

#Gallons per hour
GUSE = 5
GUSTD = .1

#Fish pricing
BASE = 1 #Expected price for worst fish
STD = .1 #Standard deviation on fish price
PERIOD = 100 #Period for type 1 fish
RATE = 1/2 #Ratio of successive type periods
SELL = .05 #Expected time until one fish is sold
SELL_STD = .005

def fish_price(typ, quality, day):
    base = BASE+(typ+quality)
    pd = PERIOD*RATE**(typ-1)
    mean = base + np.sin(day*(2*np.pi)/pd)
    return np.random.normal(mean,FSTD)

def gas_cost(day):
    return GAS_MEAN+np.random.normal(np.sin(day*(2*np.pi)/GAS_PERIOD),GAS_STD)

def gas_use(hrs):
    return sum(np.random.normal(GUSE,GUSTD,int(hrs)))+(hrs-int(hrs))*np.random.normal(GUSE,GUSTD)

def transition(hours,day):
    global fish
    fish = np.random.multinomial(abs(np.round(count(hours))),P)
    qualities = [np.random.uniform(0,1,fish[i]) for i in range(TYPES)]
    return reward(hours,day,fish,qualities)

def reward(hours,day,fish,qualities):
    reward = -gas_cost(day)*gas_use(hours)
    fsell = fish.copy()
    time_remaining = 12-hours
    while time_remaining > 0 and sum(fsell) > 0:
        time_remaining -= np.random.normal(SELL,SELL_STD)
        sell_dist = np.ceil(fsell/sum(fsell))
        sold = np.where(np.random.multinomial(1,sell_dist/np.count_nonzero(sell_dist))==1)[0][0]
        fsell[sold] -= 1
        qual = qualities[sold][-1]
        qualities[sold] = qualities[sold][:-1]
        reward += fish_price(sold+1,qual,day)
    return reward

class FishEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.observation_space = spaces.Discrete(DAYS)
        self.action_space = spaces.Discrete(12*DELTA)
        self.time = 0

    def reset(self):
        self.time = 0
        return self.time

    def step(self, action):
        self.time += 1
        return self.time, transition(action*1.0/DELTA,self.time), self.time==DAYS, {}

    def render(self, mode='human', close='False'):
        return fish
