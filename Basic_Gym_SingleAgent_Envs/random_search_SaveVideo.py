import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

def random_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, param):
    observation = env.reset()

    t = 0
    done = False
    while not done and t < 10000:
        # env.render()
        t += 1
        action = random_action(observation, param)
        observation, reward, done, _ = env.step(action)
        if done:
            break
    return t


def play_multiple_games(env, n, param):
    game_repeats = []
    for i in range(n):
        game_repeats.append(play_one_episode(env, param))

    return np.mean(game_repeats)

def search(env):
    episode_len = []
    max_len = 0

    for j in range(100):
        param = np.random.random(4)*2-1
        meanVal = play_multiple_games(env, 100, param)
        print("meanval episode {} is {}".format(j, meanVal))
        episode_len.append(meanVal)
        if meanVal > max_len:
            max_len = meanVal

    return episode_len, max_len


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'saved_videos')
    episodesLength, maxLen = search(env)
    plt.plot(episodesLength)
    plt.show()
    print("Max length: ", maxLen)