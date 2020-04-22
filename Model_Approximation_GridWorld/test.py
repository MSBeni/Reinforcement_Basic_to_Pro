import numpy as np
from TemporalDifference_TD_GridWorld.sarsa import Alpha, Gamma, All_Possible_Action

class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / 5


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
    def predict(self, s, a):
        x = self.sa2x(s, a)
        return np.dot(self.theta, x)

    def grad(self, s, a):
        return self.sa2x(s, a)

def getQs(model, s):
    # we need Q(s,a) to choose an action
    # i.e. a = argmax[a]{ Q(s,a) }
    Qs = {}
    for a in All_Possible_Action:
        q_sa = model.predict(s,a)
        Qs[a] = q_sa
    return Qs



model = Model()
s2 = (0,2)
for a in range(100):
    print(model.theta)
    print(model.predict(s2, 'U'))

    Qs2 = getQs(model, s2)
    print(Qs2)