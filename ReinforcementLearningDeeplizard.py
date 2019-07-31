"""
Tutorial on reinforcement learning by Deeplizard
https://www.youtube.com/watch?v=nyjbcRQ-uQ8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=1
"""

""" Markov Decision Processes (MDP)
A decision maker, agent, will take some action in an enviroment based on what it can observe, the state.
The action will make a change on the environment and hence enter a  new state. The action particular
action will have a corresponding reward based on the desired outcome of the model.
The state -> action -> reward cycle happens sequentialy, repeatedly. The agent always seeks to maximize 
the accumical reward over time.

We have a set of states S, actions A and rewards R. Assume these sets are finite. Time steps are denoted
by t, where t belongs to the natural numbers staring at zero.
For every S_t from the set S an action A_t is choosen from the set A. The environment is transitioned into
a new state S_(t+1). The agent receives a reward R_(t+1) from the set R based on the state-action pair.

f(S_t, A_t) = R_(t+1)

This gives the sequential process:
S_0, A_0, R_1, S_1, A_1, R_2...
"""

""" Expected return
Return (G) denotes all future rewards.
Given that we have a finite amount of steps, the agent will act in what is called an episode. This episode
runs independent of all the other episodes and is ended when a terminal state is reached. These forms of
tasks are called episodic tasks. These are called continuing tasks. This makes the return hard to define.
to make the reward converge we can use a discount on the rewards and make the agents objective to maximize
the expected discounted return of rewards. The discount rate will be a number between 0 and 1.
The discount will be a exponential of the discount rate.
"""

""" Policies and value functions
Policies (pi) maps the probability of actions to a given state. pi(a|s) is to probability of choosing a
given state s.
"""