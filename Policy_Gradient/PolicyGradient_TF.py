# https://github.com/lazyprogrammer/machine_learning_examples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import os
import sys
import datetime
from gym import wrappers
from Policy_Gradient.q_learning_beans import plot_running_avg


class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


# approximates pi(a | s)
# D: input_size --------- K: output size


# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # create the graph
    # K = number of actions
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
    layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    p_a_given_s = Z
    # action_scores = Z
    # p_a_given_s = tf.nn.softmax(action_scores)
    # self.action_scores = action_scores
    self.predict_op = p_a_given_s

    # self.one_hot_actions = tf.one_hot(self.actions, K)

    selected_probs = tf.log(
      tf.reduce_sum(
        p_a_given_s * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )
    )

    # self.selected_probs = selected_probs
    cost = -tf.reduce_sum(self.advantages * selected_probs)
    # self.cost = cost
    # self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
    self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
    # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)


# class PolicyModel:
#   def __init__(self, D, K, hidden_layer_sizes):
#     # create the graph
#     # K = number of actions
#     self.layers = []
#     M1 = D
#     for M2 in hidden_layer_sizes:
#       layer = HiddenLayer(M1, M2)
#       self.layers.append(layer)
#       M1 = M2
#
#     # final layer
#     # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
#     layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
#     self.layers.append(layer)
#
#     # inputs and targets
#     self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
#     self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
#     self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
#
#     # calculate output and cost
#     Z = self.X
#     for layer in self.layers:
#       Z = layer.forward(Z)
#     p_a_given_s = Z
#     # action_scores = Z
#     # p_a_given_s = tf.nn.softmax(action_scores)
#     # self.action_scores = action_scores
#     self.predict_op = p_a_given_s
#
#     # self.one_hot_actions = tf.one_hot(self.actions, K)
#
#     selected_probs = tf.log(
#       tf.reduce_sum(
#         p_a_given_s * tf.one_hot(self.actions, K),
#         reduction_indices=[1]
#       )
#     )











# approximate V(s)
class ValueModel:
    def __init__(self, D, hidden_layer_sizes):
        # create the graph
        self.layers = []
        M1 = D

        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x:x)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])     # the output
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))

        self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)
        # self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(10e-3, momentum=0.9).minimize(cost)


    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict = {self.X:X, self.Y:Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})



def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)

        V_next = vmodel.predict(observation)
        G = reward + gamma * np.max(V_next)
        advantage = G - vmodel.predict(prev_obs)
        pmodel.partial_fit(prev_obs, action, advantage)
        vmodel.partial_fit(prev_obs, G)

        if reward == 1:
            totalreward += reward

        iters += 1

    return totalreward


def play_one_mc(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    states = []
    actions = []
    rewards = []

    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)

        if done:
            reward = -200

        states.append(prev_obs)
        actions.append(action)
        rewards.append(reward)

        if reward == 1:
            totalreward += reward
        iters += 1

    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - vmodel.predict(s)[0])
        G = r + gamma * G

    returns.reverse()
    advantages.reverse()

    # update the model
    pmodel.partial_fit(states, actions, advantages)
    vmodel.partial_fit(states, returns)


    return totalreward



def main():
    env = gym.make('CartPole-v0')
    D = env.observation_space.shape[0]
    K = env.action_space.n
    pmodel = PolicyModel(D, K, [])
    vmodel = ValueModel(D, [10])
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.99
    #
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    totalrewards = np.empty(N)
    cost = np.empty(N)
    for n in range(N):
        totalreward = play_one_mc(env, pmodel, vmodel, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "avg reward (last 100):",
                  totalrewards[max(0, n - 100):(n + 1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()