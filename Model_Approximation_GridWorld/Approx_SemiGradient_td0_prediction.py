import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values
from TemporalDifference_TD_GridWorld.TD0_Prediction import play_game, Small_Enough, Gamma, Alpha, All_Possible_Action

# Only the policy evaluation not the optimization

# feature transformation class
class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2    # sth like [-0.37917755  0.61300533  1.40867775  0.05345664]

    def s2x(self, s):
        return np.array([s[0]-1, s[1]-1.5, s[0]*s[1]-3, 1])
        # e.g. if s = (2,0) then s2x(s) = [ 1.  -1.5 -3.   1. ]

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)    # e.g. the return will be -1.0533414541688708

    def grad(self, s):
        return self.s2x(s)


if __name__ == "__main__":

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

    model = Model()
    deltas = []
    t = 1.0
    for ti in range(20000):
        if ti % 5000 == 0:
            print(ti)
        if ti % 10 == 0:
            t += 0.01
        alpha = Alpha/t
        threshold = 0

        states_and_rewards = play_game(grid, policy)

        for n in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[n]
            s2, r = states_and_rewards[n+1]
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + Gamma * model.predict(s2)
            model.theta += alpha * (target - model.predict(s)) * model.grad(s)
            threshold = max(threshold, np.abs(old_theta - model.theta).sum())
        deltas.append(threshold)

    plt.plot(deltas)
    plt.show()

    # obtain prediction values
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0


    print("values: ")
    print_values(V, grid)
    print("policy: ")
    print_policy(policy, grid)





# s = (2,0)
#
# x = np.array([s[0]-1, s[1]-1.5, s[0]*s[1]-3, 1])
#
# theta = np.random.randn(4) / 2
#
# print(theta.dot(x))