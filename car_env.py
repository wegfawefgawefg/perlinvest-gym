import random

import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np


class PerlinvestGym:
    def __init__(self, dry_run=False):
        ''' actions:    continuous
            #   amount
            #   buy
            #   sell
            #   hold
        '''

        ''' observation:    
            #   current money
            #   current vested amount
            #   latest price
        '''
        self.STARTING_MONEY = 100
        self.action_space = (4,)
        self.observation_space = (3,)
        self.max_episode_length = 50

        self.reset()

    def get_random_state(self):
        fake_cam_frame = np.random.rand(*self.observation_space)
        return fake_cam_frame

    def step(self, action):
        self.step_count += 1
        if not self.dry_run:
            if action == 0:     #   forward
                forward()
                # self.go(self.speed)
            elif action == 1:   #   back
                backward()
                # self.go(self.speed)
            elif action == 2:   #   turn left
                slow_turn_left()
                # self.turn(-self.turn_speed)        
            elif action == 3:   #   turn right
                slow_turn_right()
                # self.turn(self.turn_speed)
            elif action == 4:
                stop()

        state_ = self.get_pic()
        reward = 0
        if self.step_count >= self.max_episode_length:
            done = True
        else:
            done = False
        info = None
        
        return state_, reward, done, info

    def get_next_next_obs(self):
        index_frac = self.step_count / self.max_episode_length
        noise =          self.noise1(index_frac)
        noise += 0.5   * self.noise2(index_frac)
        noise += 0.25  * self.noise3(index_frac)
        noise += 0.125 * self.noise4(index_frac)

        return noise

    def reset(self):
        self.step_count = 0

        self.vested = 0
        self.money = self.STARTING_MONEY

        seed = random.random() * 10
        self.noise1 = PerlinNoise(octaves=3, seed=seed)
        self.noise2 = PerlinNoise(octaves=6, seed=seed)
        self.noise3 = PerlinNoise(octaves=12, seed=seed)
        self.noise4 = PerlinNoise(octaves=24, seed=seed)

        return self.get_next_next_obs()

    def render(self):
        pass
