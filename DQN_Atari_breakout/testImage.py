import gym
# import atari_py
import matplotlib.pyplot as plt

env = gym.make("Breakout-v0")
A = env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())