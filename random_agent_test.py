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

        action = np.random.random(3)
        state, reward, done, info = env.step(action)

        num_samples += 1
        score += reward
        frame += 1

    high_score = max(high_score, score)

    print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
        num_samples, episode, high_score, score))

    episode += 1