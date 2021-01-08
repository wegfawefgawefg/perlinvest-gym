import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np


noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)
noise4 = PerlinNoise(octaves=24)

NUM_SAMPLES = 1000
xs = np.linspace(0, NUM_SAMPLES, NUM_SAMPLES)

ys = []
for i in range(NUM_SAMPLES):
    index_frac = i/NUM_SAMPLES
    noise =          noise1(index_frac)
    noise += 0.5   * noise2(index_frac)
    noise += 0.25  * noise3(index_frac)
    noise += 0.125 * noise4(index_frac)

    ys.append(noise)

plt.plot(xs, ys)
plt.show()