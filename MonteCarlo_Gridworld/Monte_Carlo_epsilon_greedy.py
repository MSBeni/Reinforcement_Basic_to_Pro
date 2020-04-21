import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values
from MonteCarlo_Gridworld.Monte_Carlo_Control import max_dict


Gamma = 0.9
All_Possible_Acions = ('U', 'D', 'L', 'R')

def random_action(a, eps=0.1):
    # A function to choose the random actions
    p = np.random.random()
    # if p<(1-eps+eps/len(All_Possible_Acions)):
    #     return a
    # else:
    #     tmp = list(All_Possible_Acions)
    #     tmp.remove(a)
    #     return np.random.choice(tmp)

    # Equal to the above code
    if p < eps:
        return a
    else:
        return np.random.choice(All_Possible_Acions)


def play_game(grid, policy):
    # returns a list of states and corresponding returns
    s = (2, 0)  # always start from this state
    grid.set_state(s)
    a = random_action(policy[s])

    # be aware of timing, each triple is s(t), a(t), r(t)
    # but r(t) results from action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]    # List of tuples of (state, action, reward)
    while True:
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])     # The next state is stochastic
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
    for t in range(5000):
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


    V = {}
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1]


    print("final values: ")
    print_values(V, grid)

    print("final policy: ")
    print_policy(policy, grid)








