import gym
import universe

env = gym.make('flashgames.DuskDrive-v0') # any Universe environment ID here
env.configure(remotes=1)
observation_n = env.reset()


while True:
  # agent which presses the Up arrow 60 times per second
  action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()

# import gym
# import universe # register Universe environments into Gym
#
# env = gym.make('flashgames.DuskDrive-v0') # any Universe [environment ID](https://github.com/openai/universe/blob/master/universe/__init__.py#L297) here
# # If using docker-machine, replace "localhost" with your Docker IP
# env.configure(remotes="vnc://localhost:5900+15900")
# env.configure(remotes=1)
# observation_n = env.reset()
#
# while True:
#   # agent which presses the Up arrow 60 times per second
#   action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
#   observation_n, reward_n, done_n, info = env.step(action_n)
#   env.render()