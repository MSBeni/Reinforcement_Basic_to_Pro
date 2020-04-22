import numpy as np
from MonteCarlo_Gridworld.gridworld import standard_grid, negative_grid
from MonteCarlo_Gridworld.gridworld import print_policy, print_values

ALL_POSSIBLE_ACTIONS = ['U','D','L','R']



def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


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

def getQs(model, s):
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s,a)
        Qs[a] = q_sa
    return Qs

def random_action(a,eps=0.3):
    draw = np.random.uniform()
    if draw < (1-eps):
        return a
    else:
        tmp = ['U','D','L','R']
        tmp.remove(a)
        action = np.random.choice(tmp)
        return action

if __name__ == '__main__':
    gamma = 0.9
    SMALL_ENOUGH = 1e-4 # threshold for convergence
    grid = negative_grid()
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

    t = 1.0
    for it in range(20000):
        if it%100 == 0:
            t += 0.001
        s = (2,0)
        Qs = getQs(model,s)
        action = max_dict(Qs)[0]
        a = random_action(action)

        grid.set_state(s)
        while not grid.game_over():
            r = grid.take_action(a)
            s2 = grid.current_state()
            Qs = getQs(model,s2)
            action = max_dict(Qs)[0]
            a2 = random_action(action, 0.3/t)
            # print s,a,s2,a2,r
            alpha = learning_rate/t
            if grid.is_terminal(s2):
                model.theta += alpha*(r - model.predict(s,a))*model.sa2x(s,a)
            else:
                Q_sa = model.predict(s2,a2)
                model.theta += alpha*(r + gamma*Q_sa - model.predict(s,a))*model.sa2x(s,a)
            s = s2
            a = a2

    for s in grid.actions.keys():
        Qs = getQs(model,s)
        policy[s], V[s] = max_dict(Qs)

    print ("Final Values:")
    print_values(V,grid)

    print ("Final Policy")
    print_policy(policy,grid)