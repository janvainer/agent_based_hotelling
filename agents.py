import random

class Agent:
    def __init__(self, actions, l)
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

    def _insert_new_state(self, state):
        """Inserts new state and its mirror image in the memoryQ
            Assigns zero payoff matrix of proper size to both new states
        """
        state = tuple(state)
        ac = self.actions
        self.memoryQ[state] = np.zeros((ac,ac))
        self.memoryQ[self._invert(state)] = np.zeros((ac,ac))
        self.memorySA[state] = np.ones((ac,ac))
        self.memorySA[self._invert(state)] = np.ones((ac,ac))
        self.memoryE[state] = 1
        self.memoryE[self._invert(state)] = 1

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


class RandomAgent(Agent):

    def next_action(self, state):
        return random.randint(self.actions), 0

    def take_response(self,my_rew,enem_rew,last_state,my_ac, enem_ac, next_state):
        # random agent ignores responses
        pass


