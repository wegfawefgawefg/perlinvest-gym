import math

import numpy as np

from perlinvest_gym import PerlinvestGym

env = PerlinvestGym()

high_score = -math.inf
episode = 0
num_samples = 0
while True:
    done = False
    state = env.reset()

    score, frame = 0, 1
    while not done:
        env.render()

        #   amount buy sell 
        #   money owned price
        amount = 0.0
        buy = 0.0
        sell = 0.0

        money, owned, price = state
        midpoint = 1.0
        if price >= midpoint:
            print("sell")
            amount = 1.0
            sell = 1.0
        elif price <= midpoint:
            print("buy")
            amount = 1.0
            buy = 1.0    
            
        action = np.array([amount, buy, sell])
        print(action)
        state_, reward, done, info = env.step(action)
        state = state_

        num_samples += 1
        score += reward
        frame += 1

    high_score = max(high_score, score)

    print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
        num_samples, episode, high_score, score))

    episode += 1