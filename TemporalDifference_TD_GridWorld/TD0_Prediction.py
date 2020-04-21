import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values

Small_Enough = 10e-4
Alpha = 0.5
Gamma = 0.9
All_Possible_Action = ['U', 'D', 'L', 'R']

def random_action(a, epsilon=0.1):
    # choosing a random action
    p = np.random.random()
    if p < epsilon:
        return a
    else:
        return np.random.choice(All_Possible_Action)


def play_game(grid, policy):

    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]    # set of tuples of state and rewards
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


if __name__ == '__main__':
    grid = standard_grid()

    print("rewards: ")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U'
    }

    # initializing the value Function
    V = {}
    all_states = grid.all_states()
    for s in all_states:
        V[s] = 0

    for iteration in range(1000):
        # defining the set of states and values to be used in the Temploral Difference
        states_and_rewards = play_game(grid, policy)

        for t in range(len(states_and_rewards) - 1):
            st, _ = states_and_rewards[t]
            st1, rt1 = states_and_rewards[t+1]

            V[st] = V[st] + Alpha * (rt1 + Gamma * V[st1] - V[st])
    print('Values: ')
    print_values(V, grid)
    print('Policy: ')
    print_policy(policy, grid)



