import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values
from TemporalDifference_TD_GridWorld.TD0_Prediction import random_action
from MonteCarlo_Gridworld.Monte_Carlo_Control import max_dict

Small_Enough = 10e-4
Alpha = 0.5
Gamma = 0.9
All_Possible_Action = ['U', 'D', 'L', 'R']

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)

    print("rewards")
    print_values(grid.rewards, grid)

    all_states = grid.all_states()

    # starting the Q value randomly
    Q = {}
    for s in all_states:
        Q[s] = {}
        for a in All_Possible_Action:
            Q[s][a] = 0

    update_counts = {}
    update_counts_s = {}
    for s in all_states:
        update_counts_s[s] = {}
        for a in All_Possible_Action:
            update_counts_s[s][a] = 1.0

    t = 1.0
    deltas = []
    for ti in range(20000):
        if ti % 100 == 0:
            t += 10e-3
        if ti % 2000 == 0:
            print('ti: ', ti)

        # initial state
        s = (2, 0)
        grid.set_state(s)

        a = max_dict(Q[s])[0]
        a = random_action(a, epsilon=0.5 / t)
        threshold = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            # We need the next action in order to calculate the Q(s,a) based on Q(s',a')
            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, epsilon=0.5 / t)     # epsilon greedy
            update_counts_s[s][a] += 0.005
            alpha = Alpha / update_counts_s[s][a]
            old_Q = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + Gamma * Q[s2][a2] - Q[s][a])
            threshold = max(threshold, np.abs(old_Q - Q[s][a]))

            # we would want to know how often Q(s) has been updated too
            update_counts[s] = update_counts.get(s, 0) + 1

            # next state becomes the current one, Checking the next state
            s = s2
            a = a2

        deltas.append(threshold)

    plt.plot(deltas)
    plt.show()

    # Determine the policy from the Q*
    # find the V* from the Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # What's the proportion of time we spent to update the Q?
    print("update counts: ")
    total = sum(update_counts.values())
    for k,v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)


    print("values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)
