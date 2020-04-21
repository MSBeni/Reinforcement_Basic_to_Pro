import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values


Gamma = 0.9
All_Possible_Acions = ('U', 'D', 'L', 'R')

def play_game(grid, policy):
    # returns a list of states and corresponding returns
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(All_Possible_Acions)  # choosing first action randomly

    # be aware of timing, each triple is s(t), a(t), r(t)
    # but r(t) results from action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]    # List of tuples of (state, action, reward)
    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        if old_s == s:
            states_actions_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            states_actions_rewards.append((s, a, r))

    # calculate the returns by working backwards from the terminal states
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it does not correspond
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + Gamma*G
    states_actions_returns.reverse()
    return states_actions_returns

def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v>max_val:
            max_val = v
            max_key = k
    return max_key, max_val


if __name__ == "__main__":
    # using the standard grid witht 0 for every state, so we can compare to
    # iterative policy evaluation
    grid = negative_grid(step_cost=-0.1)

    print("rewrds")
    print_values(grid.rewards, grid)

    # state -> action
    # initialize the random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(All_Possible_Acions)

    # initialize Q(s,a) and returns
    Q = {}
    returns = {}  # Dictionary of state -> list of returns we've received
    states = grid.all_states()
    for s in states:
        if s in grid.actions:    # not a terminal state
            Q[s] = {}
            for a in All_Possible_Acions:
                Q[s][a] = 0   # needs to be initialized to something so we can argmax
                returns[(s, a)] = []
        else:
            # terminal states or the states that we cannot enter
            pass

    # Monte_Carlo Loop
    deltas = []
    for t in range(2000):
        if t % 1000 == 0:
            print(t)
        # generate an episode using pi
        threshold = 0
        states_actions_returns = play_game(grid, policy)
        seen_states_action_pairs = set()
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called "first-visit" MC policy iteration
            sa = (s, a)
            if sa not in seen_states_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                threshold = max(threshold, np.abs(old_q - Q[s][a]))
                seen_states_action_pairs.add(sa)
        deltas.append(threshold)

        # update policy
        for a in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print("final policy: ")
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
    print("final values: ")
    print_values(V, grid)








