import gym
import numpy as np
from keras import models

env = gym.make('MountainCar-v0')
NumWins = 0
#play 20 times
#load the network
model=models.load_model('h5filetrains/trainNetworkInEPS1060.h5')

for i_episode in range(10):
    currentState = env.reset().reshape(1, 2)

    print("============================================")

    rewardSum=0
    for t in range(200):
        env.render()
        action = np.argmax(model.predict(currentState)[0])

        new_state, reward, done, info = env.step(action)

        new_state = new_state.reshape(1, 2)

        currentState=new_state

        rewardSum+=reward
        if done:
            if rewardSum > -200:
                NumWins += 1
            print("Episode finished after {} timesteps reward is {}".format(t+1,rewardSum))
            break

print(NumWins)
