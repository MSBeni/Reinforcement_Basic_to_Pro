import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values


Small_Enough = 10e-4
Gamma = 0.9
All_Possible_Acions = ('U', 'D', 'L', 'R')


def random_action(a):
    p = np.random.random()
    if p<0.5:
        return a
    else:
        tmp = list(All_Possible_Acions)
        tmp.remove(a)
        return np.random.choice(tmp)

def play_game(grid, policy):
    # returns a list of states and corresponding returns
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]    # List of tuples of (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    # calculate the returns by working backwards from the terminal states
    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it does not correspond
        if first:
            first = False
        else:
            states_and_returns.append((s,G))
        G = r + Gamma*G
    states_and_returns.reverse()
    return states_and_returns

if __name__ == "__main__":
    # using the standard grid witht 0 for every state, so we can compare to
    # iterative policy evaluation
    grid = standard_grid()

    print("rewrds")
    print_values(grid.rewards, grid)

    # state -> action
    # found by policy_iteration_random on standard_grid
    # MC method won't get exactly this, but should be close
    # values:
    # ------------------------------------
    # | 0.81 | 0.90 | 1.00  | 0.00 |
    # ------------------------------------
    # | 0.73 | 0.00 | -1.00 | 0.00 |
    # ------------------------------------
    # | 0.66 | -0.81 | -0.90| -1.00|
    # policy:
    # --------------------------------
    # | R | R | R |   |
    # --------------------------------
    # | U |   | R |   |
    # --------------------------------
    # | U | R | R | U |
# state -> action
    policy = {
      (2, 0): 'U',
      (1, 0): 'U',
      (0, 0): 'R',
      (0, 1): 'R',
      (0, 2): 'R',
      (1, 2): 'U',
      (2, 1): 'L',
      (2, 2): 'U',
      (2, 3): 'L'
    }

    # initialize V(s) and returns
    V = {}
    returns = {}  # Dictionary of state -> list of retuens we've received
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            # terminal states or the states that we cannot enter
            V[s] = 0

    # Monte_Carlo Loop
    for t in range(5000):
        # generate an episode using pi
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            # check if we have already seen s
            # called "first-visit" MC policy iteration
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)

    print("values: ")
    print_values(V, grid)
    print("policy: ")
    print_policy(policy, grid)







