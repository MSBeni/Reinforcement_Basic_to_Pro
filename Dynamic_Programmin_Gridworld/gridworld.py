import numpy as np
import matplotlib.pyplot as plt


class Grid: # Environment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # rewards should be a dict of: (i,j): r (row, col): reward
        # actions should be a dict of: (i,j): A (row, col): List of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            if action == 'D':
                self.i += 1
            if action == 'R':
                self.j += 1
            if action == 'L':
                self.j -= 1
        # returning a reward if any
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # Opposite to the move action
        if action == 'U':
            self.i += 1
        if action == 'D':
            self.i -= 1
        if action == 'R':
            self.j -= 1
        if action == 'L':
            self.j += 1
        # returning a reward if any
        assert (self.current_state() in self.all_states())


    def game_over(self):
        # return true if the game is over and false if it's not
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # simple way to get all states
        # either the position holding the possible next actions or a position that yields a reward
        return set(self.rewards.keys())|set(self.actions.keys())



def standard_grid():
    # define a grid describing the reward for arriving at each state and possible actions at each state
    # the grid is like this -- x: means cannot go there, s: start state
    #  .  .  .  1
    #  .  x  . -1
    #  s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g

def grid_world():
    grid = Grid(3, 4, [2, 0])
    rewards = {(0, 0): 0,
               (0, 1): 0,
               (0, 2): 0,
               (0, 3): 1,
               (1, 0): 0,
               (1, 2): 0,
               (1, 3): -1,
               (2, 0): 0,
               (2, 1): 0,
               (2, 2): 0,
               (2, 3): 0}

    actions = {(0, 0): ['R', 'D'],
               (0, 1): ['R', 'L'],
               (0, 2): ['R', 'L', 'D'],
               (1, 0): ['U', 'D'],
               (1, 2): ['U', 'R', 'D'],
               (2, 0): ['U', 'R'],
               (2, 1): ['L', 'R'],
               (2, 2): ['L', 'R', 'U'],
               (2, 3): ['L', 'U']}

    grid.set(rewards, actions)
    return grid

def negative_grid(step_cost = -0.1):
    grid = grid_world()
    grid.rewards.update({(0, 0): step_cost,
                         (0, 1): step_cost,
                         (0, 2): step_cost,
                         (1, 0): step_cost,
                         (1, 2): step_cost,
                         (2, 0): step_cost,
                         (2, 1): step_cost,
                         (2, 2): step_cost,
                         (2, 3): step_cost})

    return grid


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


def paly_game(agent, env):
    pass