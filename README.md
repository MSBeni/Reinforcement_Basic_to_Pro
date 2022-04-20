# RL_single_Multi_Agent_Algorithms_and_Environments
A repository to check for some funny algorithms in RL

Checking for basic to advanced RL algorithms and useful platform
Algorithms like: MonteCarlo, Dynamic_Programming, Temporal Difference, Model approximation, Policy Gradient, 
Deep Q-Network, A3C and Multi_Model RL platforms

Many different enviroment are also modeled or discussed based on this algorithms, from gym basic single agent envs 
like mountain_car to the some more complicated Universe models and also some self defined multi agent platform.

## RL Concepts
One major resource for the RL Learning path is the ```David Silver``` RL videos and docs. The Concepts below are 
extracted from his [slides](https://www.davidsilver.uk/teaching/).

### Major Components of an RL Agent
An RL agent may include one or more of these components:
- **Policy**: agent’s behaviour function
- **Value function**: how good is each state and/or action
- **Model**: agent’s representation of the environment

### Policy
A policy is the agent’s behaviour
- It is a map from state to action, e.g.
    - Deterministic policy: a = π(s)
    - Stochastic policy: π(a|s) = P[At = a|St = s]