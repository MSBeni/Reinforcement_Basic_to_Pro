import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('MountainCar-v0')
env.reset()
goal_steps = 200
score_requirements = -198
initial_game = 5000


def model_data_preparation():
    # initializing training_data and accepted_scores arrays.
    training_data = []
    accepted_scores = []
    '''
    We need to play multiple times so that we can collect the data which we can use further. 
    So we will play 10000 times so that we get a decent amount of data. This line for that “for game_index in range
    (intial_games):”
    '''
    for game_index in range(initial_game):
        '''
        We initialized score, game_memory, previous_observation variables where will store the 
        current game’s total score and previous step observation(means the position of Car and 
        its velocity) and the action we took for that
        '''
        score = 0
        game_memory = []
        previous_observation = []

        '''
        for step_index in range(goal_steps): The aim is to play the game for 200 steps 
        because episode ends when you reach 0.5(top) position, or if 200 iterations are reached.
        '''
        for step_index in range(goal_steps):
            '''
            We need to take random actions so that we can play the game which may lead to successfully 
            completing the step or losing the game. Here only 3 actions allowed push
            left(0), no push(1) and push right(2). So this code(random.randrange(0, 3)) 
            is for taking one of the random action
            '''
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            '''
            We will take that action/step. Then we will check if it’s not a first action/step 
            then we will store the previous observation and action we took for that.
            '''
            if len(previous_observation) > 0:
                game_memory.append([observation, action])

            previous_observation = observation

            '''
            Then we will check whether the position of the car which is observation[0] is greater 
            than -0.2 if yes then instead of taking the reward given by our game environment I 
            took as 1 because -0.2 position is top of the hill which means our random actions 
            giving somewhat fruitful results.
            '''
            if observation[0] > -0.2:
                reward = 1

            '''Add reward to the score and check whether the game is completed or not if yes 
            then stop playing it.'''
            score += reward

            if done:
                break

        '''
        We will check whether this game fulfilling our minimum requirement or not means are 
        we able to got score more than or equal to -198 or not.
        '''
        if score >= score_requirements:   # score_requirement = -198
            '''
            If we are able to get the score greater than or equal to -198 then we will add 
            this score to accept_scores which we further print to know how many games data 
            and their score which we are feeding to our model.
            '''
            accepted_scores.append(score)
            '''
            Then we will do hot encoding of action because its values 0(push left), 1(no push), 2(push right) 
            represent categorical data.
            '''
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                '''Then we will add that to our training_data.'''
                training_data.append([data[0], output])
        '''We will reset the environment to make sure everything clear to start playing next game.'''
        env.reset()
    '''to know how many games data and their score which we are feeding 
    to our model. Then we will return the training data.'''
    print(accepted_scores)

    return training_data

training_data = model_data_preparation()

#######################Building_The_Deep_Model#####################################################################
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))

    model.fit(X, y, epochs=10)
    return model

trained_model = train_model(training_data)

scores = []
choices = []
for each_game in range(50):
    score = 0
    game_memory = []
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices),
                                                    choices.count(2) / len(choices)))