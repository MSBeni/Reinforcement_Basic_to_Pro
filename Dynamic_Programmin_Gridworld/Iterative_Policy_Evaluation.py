import numpy as np
import matplotlib.pyplot as plt
from Dynamic_Programmin_Gridworld.gridworld import standard_grid

Small_Enough = 10e-4  # threshold for convergence

def print_values(V, g):
    for i in range(g.width):
        print("------------------------------------")
        for j in range(g.height):
            v = V.get((i,j), 0)
            if v >= 0:
                print(" |", "%.2f" %v, end="")
            else:
                print("|", "%.2f" %v, end="")
        print("  |  ")


def print_policy(P, g):
    for i in range(g.width):
        print("--------------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  |  ", a, end="")

        print("  |  ")

#######################################################

if __name__ == '__main__':
    grid = standard_grid()

    # having all the states --- all the action and reward dict keys
    states = grid.all_states()
    ## Uniformly Random Actions ##
    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0


    gamma = 1.0       # Discount Factor
    # repeat until convergence
    while True:
        threshold = 0
        for s in states:
            old_v = V[s]

            # V(s) has only the value is it is not the terminal state

            if s in grid.actions:
                new_v = 0   # we will accumulate the answer
                p_a = 1.0 / len(grid.actions[s]) # equal probability for each action
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                threshold = max(threshold, np.abs(old_v-V[s]))
        if threshold < Small_Enough:
            break
    print("Value for uniformly random actions:")
    print_values(V, grid)
    print("\n\n")

### fixed policy ###
policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
}
print_policy(policy, grid)

## initialize V(s) = 0
V = {}
for s in states:
    V[s] = 0

# let's see how V(s) changes as we get further away from the reward
gamma = 0.9

# repeat until convergence
while True:
    threshold = 0
    for s in states:
        old_v = V[s]

        # V(s) has only the value is it is not the terminal state
        if s in policy:
            a = policy[s]
            grid.set_state(s)
            r = grid.move(a)
            V[s] = r + gamma * V[grid.current_state()]
            threshold = max(threshold, np.abs(old_v-V[s]))

    if threshold < Small_Enough:
        break

print("about to finalize")
print("Value for fixed policy:")
print_values(V, grid)


