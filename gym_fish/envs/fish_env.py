import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.stats import norm

def trawl(depth):
    if SBDEPTH:
        return np.random.normal(TRAWL+DV*np.sin(2*np.pi*depth), FSTD)
    return np.random.normal(TRAWL, FSTD)

def fPMF(depth):
    raw = np.array([norm.cdf(depth+RADIUS, loc=MEANS[i], scale=STDS[i]) - norm.cdf(depth-RADIUS, loc=MEANS[i], scale=STDS[i]) for i in range(TYPES)])
    return raw/np.sum(raw)

DISCRETIZE = True #Discretize action space
DELTA = 100

DAYS = 365
TYPES = 5 #1 indexed

#Action space <depth> ranges from 0 to 1

#Fish and STD on trawl
TRAWL = 30
FSTD = 2

#Trawl radius
RADIUS = .05

#Fish locations
MEANS = np.array([i*1.0/(TYPES-1) for i in range(TYPES)])
STDS = np.array([1.0/(6*(TYPES-1)) for i in range(TYPES)])

SBDEPTH = False
DV = 5

#Pricing

#Fish pricing
BASE = 1 #Worst possible fish price
MAX = 3 #Best possible fish price
PERIOD = 365.0 #Time for pricing cycle to repeat itself
K = .2 #Spread factor

def prices(day):
    inp = .5 - .5*np.cos((2*np.pi)*day/PERIOD)
    raw = np.array([norm.cdf(inp+K, loc=MEANS[i], scale=STDS[i]) - norm.cdf(inp-K, loc=MEANS[i], scale=STDS[i]) for i in range(TYPES)])
    return BASE + (MAX-BASE)*raw/np.sum(raw)

def transition(depth,day):
    global fish
    fish = np.random.multinomial(abs(np.round(trawl(depth))),fPMF(depth))
    return fish

def reward(fish,day):
    p = prices(day)
    return sum([p[i]*fish[i] for i in range(TYPES)])

class FishEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        arr = [spaces.Discrete(TRAWL+5*FSTD) for i in range(TYPES)]
        self.observation_space = spaces.Tuple(tuple(arr))

        if DISCRETIZE:
            self.action_space = spaces.Discrete(DELTA)
        else:
            self.action_space= spaces.Box(low=np.array([0]), high=np.array([1]), dtype = np.float32)
        self.time = 0

    def reset(self):
        self.time = 0
        arr = [0 for i in range(TYPES)]
        return np.array(arr)

    def initialize(discretize=DISCRETIZE, delta=DELTA, days=DAYS, types=TYPES, trawl=TRAWL, fstd=FSTD,\
                   radius=RADIUS, means=MEANS, stds=STDS, base=BASE, mx=MAX, period=PERIOD, k=K, sbdepth=SBDEPTH, dv=DV):
        global DISCRETIZE, DELTA, DAYS, TYPES, TRAWL, FSTD, RADIUS, MEANS, STDS, BASE, MAX, PERIOD, K, SBDEPTH, DV
        DISCRETIZE = discretize
        DELTA = delta
        DAYS = days
        TYPES = types
        TRAWL = trawl
        FSTD = FSTD
        RADIUS = radius
        MEANS = means
        STDS = stds
        BASE = base
        MAX = mx
        PERIOD = period
        K = k
        SBDEPTH=sbdepth
        DV=dv

    def step(self, action):
        action = action[0]
        self.time += 1
        if DISCRETIZE:
            act = action*1.0/DELTA
        else:
            act = action
        return transition(act,self.time), reward(fish, self.time), self.time==DAYS-1, {}

    def render(self, mode='human', close='False'):
        global fish
        print(fish,self.time)
