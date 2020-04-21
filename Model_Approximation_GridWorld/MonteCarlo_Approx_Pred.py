import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values
from TemporalDifference_TD_GridWorld.TD0_Prediction import random_action
from MonteCarlo_Gridworld.Random_Monte_Carlo import random_action, play_game, Small_Enough, Gamma, All_Possible_Acions


LEARNIN_RATE = 0.001

if __name__ == "__main__":

    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'U',
        (2,1): 'L',
        (2,2): 'U',
        (2,3): 'L'
    }

    # initialize theta
    # our model is V_hat = theta.dot(x) where x=[row, col, rpw*col, 1] - 1 for bias term
    theta = np.random.randn(4)/2
    # print(theta)
    def s2x(s):
        return np.array([s[0]-1, s[1]-1.5, s[0]*s[1]-3, 1])


    # print("############TEST#############")
    # print(theta)
    # z = (2,1)
    # print(z[0], z[1])
    # l = s2x((2,0))
    # print(l)
    # print(theta.dot(l))
    # V_hat = theta.dot(l)
    # theta2 = theta
    # theta2 = theta2 + (LEARNIN_RATE/0.01) * (1.15 - V_hat) * l
    # print(theta2)
    # print(np.abs(theta - theta2).sum())
    # print(max(0, np.abs(theta2 - theta).sum()))
    # print("#############FINISH############")

    # repeat until convergence
    deltas = []
    t = 1.0
    for it in range(20000):
        if it % 100 == 0:
            t += 0.01

        alpha = LEARNIN_RATE/t
        # generate an episode using pi
        threshold = 0
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)

                theta += alpha * (G - V_hat)*x
                threshold = max(threshold, np.abs(old_theta - theta).sum())
                seen_states.add(s)
        deltas.append(threshold)

    plt.plot(deltas)
    plt.show()

    V = {}
    status = grid.all_states()
    for s in status:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:
            V[s] = 0


    print("values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)
