import numpy as np
import matplotlib.pyplot as plt
from Dynamic_Programmin_Gridworld.gridworld import standard_grid, negative_grid
from Dynamic_Programmin_Gridworld.gridworld import print_policy, print_values

Small_Enough = 10e-4
Gamma = 0.9
All_Possible_Acions = ('U', 'D', 'L', 'R')

# this id deterministic
# all p(s',r|s,a) = 1 or 0

if __name__ == "__main__":
    # this grid will give you a reward of -0.1 every non-terminal state
    # we want to see if this encourage a shorter path to the gaol
    grid = negative_grid(step_cost=-1.0)

    # print rewards
    print_values(grid.rewards, grid)


    # state -> action
    # we will randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(All_Possible_Acions)

    # initial policy
    print("initial policy:")
    print_policy(policy, grid)


    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = o
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal states
            V[s] = 0

    while True:
        # policy evaluation step
        while True:
            threshold = 0
            for s in states:
                v_old = V[s]

                # V[s] only has value only if it is not a terminal state
                v_new = 0
                if s in policy:
                    for a in All_Possible_Acions:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3

                        grid.set_state(s)
                        r = grid.move(a)
                        v_new = p * (r + Gamma * V[grid.current_state()])
                    V[s] = v_new
                    threshold = max(threshold, np.abs(v_old - V[s]))
            if threshold < Small_Enough:
                break
        # policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                # loop through all possible action to find the best one
                for a in All_Possible_Acions:  # chosen action
                    v = 0
                    for a2 in All_Possible_Acions:   # resulting action
                        if a2 == a:
                            p = 0.5
                        else:
                            p = 0.5/3

                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p * (r + Gamma * V[grid.current_state()])
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a!=old_a:
                    is_policy_converged = False
        if is_policy_converged:
            break

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)