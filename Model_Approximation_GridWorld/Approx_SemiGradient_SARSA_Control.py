
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values
from MonteCarlo_Gridworld.Monte_Carlo_Control import max_dict
from TemporalDifference_TD_GridWorld.sarsa import Alpha, Gamma, All_Possible_Action


SA2IDX = {}
IDX = 0

class Model:
    def __init__(self):
        self.theta = np.random.randn(25)/5

    def sa2x(self,s,a):
        return np.array([
        s[0] - 1               if a == 'U' else 0,
        s[1] - 1.5             if a == 'U' else 0,
        (s[0]*s[1] - 3)/3      if a == 'U' else 0,
        (s[0]*s[0] - 2)/2      if a == 'U' else 0,
        (s[1]*s[1] - 4.5)/4.5  if a == 'U' else 0,
        1                      if a == 'U' else 0,
        s[0] - 1               if a == 'D' else 0,
        s[1] - 1.5             if a == 'D' else 0,
        (s[0]*s[1] - 3)/3      if a == 'D' else 0,
        (s[0]*s[0] - 2)/2      if a == 'D' else 0,
        (s[1]*s[1] - 4.5)/4.5  if a == 'D' else 0,
        1                      if a == 'D' else 0,
        s[0] - 1               if a == 'R' else 0,
        s[1] - 1.5             if a == 'R' else 0,
        (s[0]*s[1] - 3)/3      if a == 'R' else 0,
        (s[0]*s[0] - 2)/2      if a == 'R' else 0,
        (s[1]*s[1] - 4.5)/4.5  if a == 'R' else 0,
        1                      if a == 'R' else 0,
        s[0] - 1               if a == 'L' else 0,
        s[1] - 1.5             if a == 'L' else 0,
        (s[0]*s[1] - 3)/3      if a == 'L' else 0,
        (s[0]*s[0] - 2)/2      if a == 'L' else 0,
        (s[1]*s[1] - 4.5)/4.5  if a == 'L' else 0,
        1                      if a == 'L' else 0,
        1
        ])

    def predict(self,state,action):
        x = self.sa2x(state,action)
        return np.dot(self.theta, x)
    def grad(self, s, a):
        return self.sa2x(s, a)

# turning the Q to the dictionary given the s and a
def getQs(model, s):
    # we need Q(s,a) to choose an action
    # i.e. a = argmax[a]{ Q(s,a) }
    Qs = {}
    for a in All_Possible_Action:
        q_sa = model.predict(s,a)
        Qs[a] = q_sa
    return Qs

def random_action(a,eps=0.3):
    draw = np.random.uniform()
    if draw < (1-eps):
        return a
    else:
        tmp = ['U', 'D', 'L', 'R']
        tmp.remove(a)
        action = np.random.choice(tmp)
        return action


if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)

    print("rewards")
    print_values(grid.rewards, grid)
    #
    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in All_Possible_Action:
            SA2IDX[s][a] = IDX
            IDX += 1
    #
    # model = Model()
    deltas = []
    #
    # t = 1.0
    # t2 = 1.0
    gamma = 0.9
    SMALL_ENOUGH = 1e-4 # threshold for convergence
    # grid = negative_grid()
    model = Model()
    V = {}
    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'R',
        (2,1): 'R',
        (2,2): 'R',
        (2,3): 'U'
    }
    learning_rate = 0.01

    states = grid.all_states()
    for s in states:
        if s not in grid.actions.keys():
            V[s] = 0

    V = {}
    policy = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    t2 = 1.0
    t = 1.0
    for ti in range(20000):
        if ti % 1000 == 0:
            print("ti: ", ti)
        if ti% 100 == 0:
            t += 0.001
            t2 += 0.01

        threshold = 0
        s = (2, 0)
        Qs = getQs(model, s)
        a = max_dict(Qs)[0]
        a = random_action(a, eps=0.5/t)

        grid.set_state(s)

        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            old_theta = model.theta.copy()

            alpha = learning_rate / t2

            if grid.is_terminal(s2):
                model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)

            else:
                Qs2 = getQs(model, s2)
                # print(Qs2)
                a2 = max_dict(Qs2)[0]
                # print(a2)
                a2 = random_action(a2, eps=0.5/t)

                model.theta += alpha * (r + Gamma * model.predict(s2, a2) - model.predict(s, a)) * model.grad(s, a)

                # next state becomes current one
                s = s2
                a = a2
            threshold = max(threshold, np.abs(old_theta - model.theta).sum())

        deltas.append(threshold)

    plt.plot(deltas)
    plt.show()

    # # Determine the policy from the Q*
    # find the V* from the Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        # print s
        # print Qs
        policy[s], V[s] = max_dict(Qs)




    print("values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)



#
# import numpy as np
#
# ALL_POSSIBLE_ACTIONS = ['U','D','L','R']
#
# class Grid:
#     def __init__(self, height, width, start):
#         self.height = height
#         self.width = width
#         self.i = start[0]
#         self.j = start[1]
#
#     def set(self, rewards, actions):
#         self.rewards = rewards
#         self.actions = actions
#
#     def set_state(self,state):
#         self.i = state[0]
#         self.j = state[1]
#
#     def current_state(self):
#         return (self.i, self.j)
#
#     def is_terminal(self, state):
#         if state not in self.actions.keys():
#             return True
#         return False
#
#     def take_action(self,action):
#         if action in self.actions[(self.i, self.j)]:
#             if action == 'U':
#                 self.i -= 1
#             elif action == 'D':
#                 self.i += 1
#             elif action == 'L':
#                 self.j -= 1
#             elif action == 'R':
#                 self.j += 1
#         else:
#             # print "Cannot perform this action"
#             return 0
#         return self.rewards[(self.i, self.j)]
#
#     def undo_move(self, action):
#         if action == 'U':
#             self.i += 1
#         elif action == 'D':
#             self.i -= 1
#         elif action == 'L':
#             self.j += 1
#         elif action == 'R':
#             self.j -= 1
#
#         assert(self.current_state() in self.all_states)
#
#     def game_over(self):
#         return (self.current_state() not in self.actions.keys())
#
#     def all_states(self):
#         return set(self.rewards.keys())|set(self.actions.keys())
#
# def grid_world():
#     grid = Grid(3,4,[2,0])
#     rewards = {(0,0):0,
#                (0,1):0,
#                (0,2):0,
#                (0,3):1,
#                (1,0):0,
#                (1,2):0,
#                (1,3):-1,
#                (2,0):0,
#                (2,1):0,
#                (2,2):0,
#                (2,3):0}
#
#     actions = {(0,0):['R','D'],
#                (0,1):['R','L'],
#                (0,2):['R','L','D'],
#                (1,0):['U','D'],
#                (1,2):['U','R','D'],
#                (2,0):['U','R'],
#                (2,1):['L','R'],
#                (2,2):['L','R','U'],
#                (2,3):['L','U']}
#
#     grid.set(rewards, actions)
#     return grid
#
# def print_values(V, g):
#     for i in range(g.width):
#         print("------------------------------------")
#         for j in range(g.height):
#             v = V.get((i,j), 0)
#             if v >= 0:
#                 print(" |", "%.2f" %v, end="")
#             else:
#                 print("|", "%.2f" %v, end="")
#         print("  |  ")
#
#
# def print_policy(P, g):
#     for i in range(g.width):
#         print("--------------------------------")
#         for j in range(g.height):
#             a = P.get((i, j), ' ')
#             print("  |  ", a, end="")
#
#         print("  |  ")
#
#
# def max_dict(d):
#     # returns the argmax (key) and max (value) from a dictionary
#     # put this into a function since we are using it so often
#     max_key = None
#     max_val = float('-inf')
#     for k, v in d.items():
#         if v > max_val:
#             max_val = v
#             max_key = k
#     return max_key, max_val
#
# def negative_grid(step_cost = -0.1):
#     grid = grid_world()
#     grid.rewards.update({(0,0):step_cost,
#                          (0,1):step_cost,
#                          (0,2):step_cost,
#                          (1,0):step_cost,
#                          (1,2):step_cost,
#                          (2,0):step_cost,
#                          (2,1):step_cost,
#                          (2,2):step_cost,
#                          (2,3):step_cost})
#
#     return grid
#
# class Model:
#     def __init__(self):
#         self.theta = np.random.randn(25)/5
#
#     def sa2x(self,s,a):
#         return np.array([
#         s[0] - 1               if a == 'U' else 0,
#         s[1] - 1.5             if a == 'U' else 0,
#         (s[0]*s[1] - 3)/3      if a == 'U' else 0,
#         (s[0]*s[0] - 2)/2      if a == 'U' else 0,
#         (s[1]*s[1] - 4.5)/4.5  if a == 'U' else 0,
#         1                      if a == 'U' else 0,
#         s[0] - 1               if a == 'D' else 0,
#         s[1] - 1.5             if a == 'D' else 0,
#         (s[0]*s[1] - 3)/3      if a == 'D' else 0,
#         (s[0]*s[0] - 2)/2      if a == 'D' else 0,
#         (s[1]*s[1] - 4.5)/4.5  if a == 'D' else 0,
#         1                      if a == 'D' else 0,
#         s[0] - 1               if a == 'R' else 0,
#         s[1] - 1.5             if a == 'R' else 0,
#         (s[0]*s[1] - 3)/3      if a == 'R' else 0,
#         (s[0]*s[0] - 2)/2      if a == 'R' else 0,
#         (s[1]*s[1] - 4.5)/4.5  if a == 'R' else 0,
#         1                      if a == 'R' else 0,
#         s[0] - 1               if a == 'L' else 0,
#         s[1] - 1.5             if a == 'L' else 0,
#         (s[0]*s[1] - 3)/3      if a == 'L' else 0,
#         (s[0]*s[0] - 2)/2      if a == 'L' else 0,
#         (s[1]*s[1] - 4.5)/4.5  if a == 'L' else 0,
#         1                      if a == 'L' else 0,
#         1
#         ])
#
#     def predict(self,state,action):
#         x = self.sa2x(state,action)
#         return np.dot(self.theta, x)
#
# def getQs(model, s):
#     Qs = {}
#     for a in ALL_POSSIBLE_ACTIONS:
#         q_sa = model.predict(s,a)
#         Qs[a] = q_sa
#     return Qs
#
# def random_action(a,eps=0.3):
#     draw = np.random.uniform()
#     if draw < (1-eps):
#         return a
#     else:
#         tmp = ['U','D','L','R']
#         tmp.remove(a)
#         action = np.random.choice(tmp)
#         return action
#
# if __name__ == '__main__':
#     gamma = 0.9
#     SMALL_ENOUGH = 1e-4 # threshold for convergence
#     grid = negative_grid()
#     model = Model()
#     V = {}
#     policy = {
#         (2,0): 'U',
#         (1,0): 'U',
#         (0,0): 'R',
#         (0,1): 'R',
#         (0,2): 'R',
#         (1,2): 'R',
#         (2,1): 'R',
#         (2,2): 'R',
#         (2,3): 'U'
#     }
#     learning_rate = 0.01
#
#     states = grid.all_states()
#     for s in states:
#         if s not in grid.actions.keys():
#             V[s] = 0
#
#     V = {}
#     policy = {}
#     states = grid.all_states()
#     for s in states:
#         V[s] = 0
#
#     t = 1.0
#     for it in range(20000):
#         if it%100 == 0:
#             t += 0.001
#         s = (2,0)
#         Qs = getQs(model,s)
#         action = max_dict(Qs)[0]
#         a = random_action(action)
#
#         grid.set_state(s)
#         while not grid.game_over():
#             r = grid.take_action(a)
#             s2 = grid.current_state()
#             Qs = getQs(model,s2)
#             action = max_dict(Qs)[0]
#             a2 = random_action(action, 0.3/t)
#             # print s,a,s2,a2,r
#             alpha = learning_rate/t
#             if grid.is_terminal(s2):
#                 model.theta += alpha*(r - model.predict(s,a))*model.sa2x(s,a)
#             else:
#                 Q_sa = model.predict(s2,a2)
#                 model.theta += alpha*(r + gamma*Q_sa - model.predict(s,a))*model.sa2x(s,a)
#             s = s2
#             a = a2
#
#     for s in grid.actions.keys():
#         Qs = getQs(model,s)
#         print(s)
#         print (Qs)
#         policy[s], V[s] = max_dict(Qs)
#
#     print("Final Values:")
#     print_values(V,grid)
#
#     print ("Final Policy")
#     print_policy(policy,grid)