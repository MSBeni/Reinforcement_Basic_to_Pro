import gym
env = gym.make('BipedalWalker-v2')
for i_episode in range(100):
    observation = env.reset()
    for t in range(10000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("{} timesteps taken for the episode".format(t + 1))
            break