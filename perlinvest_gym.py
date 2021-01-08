import random

import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np

'''TODO:
trades
open interest
volume
'''

class PerlinvestGym:
    def __init__(self, verbose=False):
        ''' actions:    continuous
            #   amount
            #   buy
            #   sell
        '''
        ''' observation:    
            #   current money
            #   current owned
            #   latest price
        '''
        self.verbose = verbose
        self.STARTING_MONEY = 100
        self.MAX_EPISODE_LENGTH = 100
        self.action_space = (3,)
        self.observation_space = (3,)

        #   defined in reset:
        self.owned = 0
        self.money = self.STARTING_MONEY
        self.price = 0
        self.noise1, self.noise2, self.noise3, self.noise4 = \
            None, None, None, None
        self.reset()

    def step(self, action):
        self.step_count += 1
        info = {}

        if self.verbose: print(f"money: {self.money:.2f}, owned: {self.owned:.2f}, price: {self.price:.2f}")

        amount, buy, sell = action
        if buy > 0.5:   #   buy priority is higher than sell
            cost = amount * self.price
            if cost <= self.money:
                self.money -= cost
                self.owned += amount
                if self.verbose: print(f"bought {amount:.2f} @ {self.price:.2f}, TOTAL: {cost:.2f}")
                if self.verbose: print(f"new money: {self.money:.2f}")
        elif sell > 0.5:
            cost = amount * self.price
            if amount <= self.owned:
                self.owned -= amount
                self.money += cost
                if self.verbose: print(f"sold {amount:.2f} @ {self.price:.2f}, TOTAL: {cost:.2f}")
                if self.verbose: print(f"new money: {self.money:.2f}")
        self.price = self.get_next_price()
        state_ = np.array([
            self.money, 
            self.owned, 
            self.price
        ])
        
        ###         EPISODE TERMINATION CONDS.      ###
        done = False
        if self.money < 50:
            done = True
        if self.step_count >= self.MAX_EPISODE_LENGTH:
            done = True

        reward = 1.0
        
        info = None
        
        return state_, reward, done, info

    def get_next_price(self):
        index_frac = self.step_count / self.MAX_EPISODE_LENGTH
        noise =          self.noise1(index_frac)
        noise += 0.5   * self.noise2(index_frac)
        noise += 0.25  * self.noise3(index_frac)
        noise += 0.125 * self.noise4(index_frac)
        noise += 1.0

        assert noise >= 0.0     #   NEGATIVE PRICE

        return noise

    def reset(self):
        self.step_count = 0

        self.owned = 0
        self.money = self.STARTING_MONEY

        seed = random.randint(1, 1000)
        self.noise1 = PerlinNoise(octaves=3, seed=seed)
        self.noise2 = PerlinNoise(octaves=6, seed=seed)
        self.noise3 = PerlinNoise(octaves=12, seed=seed)
        self.noise4 = PerlinNoise(octaves=24, seed=seed)

        self.price = self.get_next_price()
        state = np.array([
            self.money, 
            self.owned, 
            self.price
        ])

        return state

    def render(self):
        pass
