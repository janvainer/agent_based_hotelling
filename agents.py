import random
import numpy as np


class Agent:
    def __init__(self, actions, l):
        """Set agent's learning parameters and available actions.

        Keyword arguments:
        l - location
        actions -- number of available actions
        """
        self.l = l
        self.actions = actions
        # learned q values
        self.memoryQ = {}
        # number of times (s,a) was visited, same construction like memoryQ
        self.memorySA = {}
        # how many times s was visited, e_t(s) = c/n 
        self.memoryE = {}
        # constant for exploration rate
        self.c = 0.2

    def reset(self):
        self.memoryQ = {}
        self.memorySA = {}
        self.memoryE = {}

    def _invert(self,state):
        """Returns mirror image of current state"""

        return (state[1],state[0])

    def _insert_new_state(self, state):
        state = tuple(state)
        ac = self.actions
        self.memoryQ[state] = np.zeros(ac)
        self.memorySA[state] = np.ones(ac)
        self.memoryE[state] = 1

    # actions must come in the order in which the agents were given at the beginning
    def take_response(self,my_rew,enem_rew,last_state,my_ac, enem_ac, next_state):
        """Update memory based on own and opponent's experience"""
        raise NotImplementedError("Reimplement this to get a valid agent")

    def next_action(self, state):
        """Returns action number as integer index and expected utility of the action

        Keyword arguments:
        state -- a tuple (a,b), where a is distance from the nearest side
        """
        raise NotImplementedError("Reimplement this to get a valid agent")

    def random_action(self):
        return np.asscalar(np.random.choice(self.actions,1)), 0


class RandomAgent(Agent):

    def next_action(self, state):
        return self.random_action()

    def take_response(self,my_rew,enem_rew,last_state,my_ac, enem_ac, next_state):
        # random agent ignores responses
        pass


class QAgent(Agent):

    def next_action(self, state):
        state = tuple(state)

        # insert new state + its invert, decide randomly upon an action
        if state not in self.memoryQ:
            self._insert_new_state(state)
            return self.random_action()

        # explore, or play Nash strategy based on agents expectations, exploration if random < c/n(s)    
        if np.random.random() < self.c/self.memoryE[state]:
            return self.random_action()
        else:
            best_action = np.argmax(self.memoryQ[state])
            expected_utility = max(self.memoryQ[state])
            return best_action, expected_utility


    def take_response(self,my_rew,enem_rew,last_state,my_ac, enem_ac, next_state):
        last_state = tuple(last_state)
        q = self.memoryQ[last_state][my_ac]
        # 1/n learning rate guarantees convergence thanks to non-zero exploration rate
        a = 1/self.memorySA[last_state][my_ac]
        l = self.l
        r = my_rew

        # Price agents are contextual bandits
        # Location agents are Q agents
        if next_state not in self.memoryQ:
            max_q = 0
        elif next_state == last_state:
                max_q = 0
        else: max_q = max(self.memoryQ[next_state])

        q = (1-a)*q + a*(r+l*max_q)
        self.memoryQ[last_state][my_ac] = q
        self.memorySA[last_state][my_ac] +=1
