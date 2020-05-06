import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from datetime import datetime
from gym.envs.mujoco import mujoco_env

# simple feedforward neural net
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
  for h in layer_sizes[:-1]:
    x = tf.layers.dense(x, units=h, activation=hidden_activation)
  return tf.layers.dense(x, units=layer_sizes[-1], activation=output_activation)
# def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
#     for h in layer_sizes[:-1]:
#         x = tf.layers.dense(x, units=h, activation=hidden_activation)
#     return tf.layers.dense(x, units=layer_sizes, activation=output_activation)

# get all variables within a scope
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


# Create both actor and critic net at once
###### Q(s, mu(s)) returns the maximum Q for a given state s ######
def CreateNetworks(
        s, a,      # s is a placeholder for state and a is a placeholder for action
        num_actions,
        action_max,    # maximum absolute value of the actions, we assume the upper and the lower band are the same
        hidden_sizes=(300,),
        hidden_activation=tf.nn.relu,
        output_activation=tf.tanh):

    with tf.variable_scope('mu'):
        mu = action_max * ANN(s, list(hidden_sizes)+[num_actions], hidden_activation, output_activation)
    # we receive the state and we have one output node per action
    with tf.variable_scope('q'):
        input_ = tf.concat([s, a], axis=1)   # (state, action)
        # since the final layer has the output of size one we get back a tensor of size (n, 1)
        # get rid of the redundant dimensions with tf.squeeze
        q = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    # the composition of the neural networks to have the Q(s, mu(s))
    with tf.variable_scope('q', reuse=True):
        # reuse is True, so it returns the weights from the previously defined Q network
        input_ = tf.concat([s, mu], axis=1)
        q_mu = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    return mu, q, q_mu

# the experience replay memory
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size


    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # check for the size and make it zero when reach the max
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


# DDPG algorithm implementation
def ddpg(
        env_fn,              # callable func returns an instance of the environment
        ac_kwargs=dict(),    # actor critic keyword argument
        seed=0,              # make our experience reproducable
        save_folder=None,
        num_train_episodes=100,
        test_agent_every=25,
        replay_size=int(1e6),
        gamma=0.99,
        decay=0.995,         # smoothing target we use, when we update the target network parameters
        mu_lr=1e-3,          # learning rate of the mu for the policy network
        q_lr=1e-3,           # learning rate of the q for the value network
        batch_size=100,
        start_steps=10000,
        action_noise=0.1,
        max_episode_length=1000):

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # Recording the video of the test
    if save_folder is not None:
        test_env = gym.wrappers.Monitor(test_env, save_folder)

    # get the size of the action space and state space
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # maximum value of the action
    # Assume both low and the high values are the same
    # Assume all the actions have the same bounds
    # May Not be the case for all the environments
    action_max = env.action_space.high[0]

    # Create Tensorflow placeholders (neural network inputs)
    X = tf.placeholder(dtype=tf.float32, shape=(None, num_states))  # state
    A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions))  # action
    X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states))  # next state
    R = tf.placeholder(dtype=tf.float32, shape=(None,))  # reward
    D = tf.placeholder(dtype=tf.float32, shape=(None,))  # done

    # main network outputs
    with tf.variable_scope('main'):
        mu, q, q_mu = CreateNetworks(X, A, num_actions, action_max, **ac_kwargs)

    # Target network
    with tf.variable_scope('target'):
        # We don't need the Q network output with arbitrary input action A
        # because that's not actually used in our loss functions
        # NOTE 1: The state input is X2, NOT X
        #         We only care about max_a{ Q(s', a) }
        #         Where this is equal to Q(s', mu(s'))
        #         This is because it's used in the target calculation: r + gamma * max_a{ Q(s',a) }
        #         Where s' = X2
        # NOTE 2: We ignore the first 2 networks for the same reason
        _, _, q_mu_targ = CreateNetworks(X2, A, num_actions, action_max, **ac_kwargs)

    # Experience replay memory
    replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)

    # Target value for the Q-network loss
    # We use stop_gradient to tell Tensorflow not to differentiate
    # q_mu_targ wrt any params
    # i.e. consider q_mu_targ values constant
    q_target = tf.stop_gradient(R + gamma * (1 - D) * q_mu_targ)

    # DDPG losses
    mu_loss = -tf.reduce_mean(q_mu)
    q_loss = tf.reduce_mean((q - q_target)**2)

    # Train each network separately
    mu_optimizer = tf.train.AdamOptimizer(learning_rate=mu_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    mu_train_op = mu_optimizer.minimize(mu_loss, var_list=get_vars('main/mu'))
    q_train_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))


    # Use soft updates to update the target network
    target_update = tf.group(
        [tf.assign(v_targ, decay*v_targ + (1 - decay)*v_main)
         for v_main, v_targ in zip(get_vars('main'), get_vars('target'))]
    )

    # Copy main network params to target networks
    target_init = tf.group(
        [tf.assign(v_targ, v_main)
         for v_main, v_targ in zip(get_vars('main'), get_vars('target'))]
    )


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(s, noise_scale):
        a = sess.run(mu, feed_dict={X: s.reshape(1, -1)})[0]
        a += noise_scale * np.random.randn(num_actions)
        return np.clip(a, -action_max, action_max)

    test_returns = []
    def test_agent(num_episides=5):
        t0 = datetime.now()
        n_steps = 0
        for j in range(num_episides):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            while not (d or (episode_length == max_episode_length)):
                # Take deterministic actions at test time (noise_scale=0)
                test_env.render()
                s, r, d, _ = test_env.step(get_action(s, 0))
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return: ', episode_return, 'episode length: ', episode_length)
            test_returns.append(episode_return)

    # Main loop: play episode and train
    returns = []
    q_losses = []
    mu_losses = []
    num_steps = 0
    for i_episode in range(num_train_episodes):

        # reset env
        s, episode_return, episode_length, d = env.reset(), 0, 0, False

        while not (d or (episode_length == max_episode_length)):
            # For the first `start_steps` steps, use randomly sampled actions
            # in order to encourage exploration.
            if num_steps > start_steps:
                a = get_action(s, action_noise)
            else:
                a = env.action_space.sample()

            # keep track of the number of the steps done
            num_steps += 1
            if num_steps == start_steps:
                print("Using Agent Action Now")


            # step the env
            s2, r, d, _ = env.step(a)
            episode_return += r
            episode_length += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d_store = False if episode_length == max_episode_length else d

            # store experience to replay buffer
            replay_buffer.store(s, a, r, s2, d_store)

            # Assign next state to be the current state on the next round
            s = s2

            # Perform the updates
            for _ in range(episode_length):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    X: batch['s'],
                    X2: batch['s2'],
                    A: batch['a'],
                    R: batch['r'],
                    D: batch['d']
                }

            # Q network update
            # Note: plot the Q loss if you want
            ql, _, _ = sess.run([q_loss, q, q_train_op], feed_dict)
            q_losses.append(ql)

            # Policy update
            # (And target networks update)
            # Note: plot the mu loss if you want
            mul, _, _ = sess.run([mu_loss, mu_train_op, target_update], feed_dict)
            mu_losses.append(mul)

        print("Episode:", i_episode+1, "  Return:", episode_return, "  episode_length:", episode_length)
        returns.append(episode_return)

        # Test the agent
        if i_episode > 0 and i_episode % test_agent_every == 0:
            test_agent()
    # on Mac, plotting results in an error, so just save the results for later
    # if you're not on Mac, feel free to uncomment the below lines
    np.savez('ddpg_results.npz', train=returns, test=test_returns, q_losses=q_losses, mu_losses=mu_losses)

    plt.plot(returns)
    plt.plot(smooth(np.array(returns)))
    plt.title("Train returns")
    plt.show()

    plt.plot(test_returns)
    plt.plot(smooth(np.array(test_returns)))
    plt.title("Test returns")
    plt.show()

    plt.plot(q_losses)
    plt.title('q_losses')
    plt.show()

    plt.plot(mu_losses)
    plt.title('mu_losses')
    plt.show()



def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hidden_layer_sizes', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_train_episodes', type=int, default=200)
    parser.add_argument('--save_folder', type=str, default='ddpg_monitor')
    args = parser.parse_args()

    ddpg(
    lambda : gym.make(args.env),
    ac_kwargs=dict(hidden_sizes=[args.hidden_layer_sizes]*args.num_layers),
    gamma=args.gamma,
    seed=args.seed,
    save_folder=args.save_folder,
    num_train_episodes=args.num_train_episodes,
    )
