import gym
import numpy as np

# Then we make our frozen lake environment using OpenAI's Gym:
env = gym.make('FrozenLake-v0')
env = env.unwrapped
# We will first explore the environments.
# The number of states in the environment is 16 as we have a 4*4 grid:

# print(env.observation_space.n)

#
# def value_iteration(env, gamma=1.0):
#     # First, we initialize the random value table which is 0 for all the states and numbers of the iterations
#     value_table = np.zeros(env.observation_space.n)
#     no_of_iterations = 100000
#     threshold = 1e-20
#     for i in range(no_of_iterations):
#         # Then, upon starting each iteration, we copy the value_table to updated_value_table
#         updated_value_table = np.copy(value_table)
# #         As we calculate next_state_rewards for all actions of a state and append it to our Q
# #           value, we pick up the maximum Q value and update it as a value of our state:
#         for state in range(env.observation_space.n):
#             Q_value = []
#             for action in range(env.action_space.n):
#                 next_states_rewards = []
#                 for next_sr in env.P[state][action]:
#                     trans_prob, next_state, reward_prob, _ = next_sr
#                     next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
#                 Q_value.append(np.sum(next_states_rewards))
#             value_table[state] = max(Q_value)
#         if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
#             print('Value-iteration converged at iteration# %d.' % (i + 1))
#             break
#     return value_table, Q_value
def value_iteration(env, gamma=1.0):
    # initialize value table with zeros
    value_table = np.zeros(env.observation_space.n)

    # set number of iterations and threshold
    no_of_iterations = 100000
    threshold = 1e-20

    for i in range(no_of_iterations):

        # On each iteration, copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)
        # print(updated_value_table)
        # print("\n")

        # Now we calculate Q Value for each actions in the state
        # and update the value of a state with maximum Q value

        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    # print(next_sr)
                    # print("\n")
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)

            # we will check whether we have reached the convergence i.e whether the difference
        # between our value table and updated value table is very small. But how do we know it is very
        # small? We set some threshold and then we will see if the difference is less
        # than our threshold, if it is less, we break the loop and return the value function as optimal
        # value function

        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break

    return value_table
# def extract_policy(value_table, gamma = 1.0):
#     policy = np.zeros(env.observation_space.n)
#     for state in range(env.observation_space.n):
#         Q_table = np.zeros(env.action_space.n)
#         for action in range(env.action_space.n):
#             for next_sr in env.P[state][action]:
#                 trans_prob, next_state, reward_prob, _ = next_sr
#                 Q_table[action] += (trans_prob * (reward_prob + gamma *value_table[next_state]))
#         policy[state] = np.argmax(Q_table)
#     return policy

def extract_policy(value_table, gamma=1.0):
    # initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):

        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)

        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

optimal_value_function = value_iteration(env=env,gamma=1.0)
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(optimal_policy)

