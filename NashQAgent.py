import nash as ns 
import numpy as np

# This module contains the NashQ agent class with core functions of the simulation
# arena has size [0,...,6]... 3 is in the middle


class NashQ:
	def __init__(self, actions, l = 0.8):
		"""Set agent's learning parameters and available actions. 
		
		Keyword arguments:
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

	# returns the memory object
	def get_memory(self):
		return self.memoryQ
		
	def next_action(self,state):
		"""Returns action number as integer index and expected utility of the action
		
		Keyword arguments:
		state -- a tuple (a,b), where a is distance from the nearest side
		"""

		# insert new state + its invert, decide randomly upon an action
		if state not in self.memoryQ:
			self._insert_new_state(state)
			return np.asscalar(np.random.choice(self.actions,1)), 0 

		# explore, or play Nash strategy based on agents expectations, exploration if random < c/n(s)	
		if np.random.random() < self.c/self.memoryE[state]:
			return np.asscalar(np.random.choice(self.actions,1)), 0
		else:
			my_nash = self._nash_q(state) # = [ [[my distr] ,[his distrib]] , [my util, his util] ]
			mixed_strategy = my_nash[0][0] # np array - my distribution
			# play the mixed equilibrium strategy
			
			# returns his NE strategy and expected utility of self agent
			return np.asscalar(np.random.choice(self.actions,1,p=mixed_strategy.tolist())), my_nash[1][0]


	# actions must come in the order in which the agents were given at the beginning
	def take_response(self,my_rew,enem_rew,last_state,my_ac, enem_ac, next_state):
		"""Update memory based on own and opponent's experience"""
			
		# learn from own feedback
		self._response(my_rew,last_state,my_ac,enem_ac,next_state)
		# invert states and actions to learn from opponents experience
		self._response(enem_rew, self._invert(last_state),enem_ac,my_ac,self._invert(next_state))
			
	def _response(self,rew,last_state, my_ac,enem_ac,next_state):	
		
		# use act to slice np.arrays in the memoryQ
		act = (my_ac,enem_ac) 
		
		q = self.memoryQ[last_state][act]
		# 1/n learning rate guarantees convergence thanks to non-zero exploration rate
		a = 1/self.memorySA[last_state][act]
		l = self.l
		r = rew

		# Price agents are contextual bandits
		# Location agents are Q agents
		if next_state not in self.memoryQ:
			nash_q = 0		
		elif next_state == last_state:
				nash_q = 0
		else: nash_q = self._nash_q(next_state)[1][0]

		q = (1-a)*q + a*(r+l*nash_q)
		self.memoryQ[last_state][act] = q
		self.memorySA[last_state][act] +=1 		
		
	def _nash_q(self,state):
		"""Returns first NE and utilities as [eq, utils]
			
			eq is a vector of numbers [a,b], where index of a is an index of an action from actions and a is a probability of playing that ction
			utils is a vector of utilities [u1,u2] for a given eq
			
			Keyword arguments:
			state -- a tuple (a,b), where a is distance from the nearest side
		"""
		
		# agent finds NE as if the opponent was himself. - He just inverts current
		# state to represent the opponent from his own memory
		q = [self.memoryQ[state],self.memoryQ[self._invert(state)]]
		game = ns.Game(q[0],q[1])
		eq = [e for e in game.support_enumeration()]
		
		# check if not degenerate. If degenerate, select random strategy
		if not eq:
			eq = [np.full((1,self.actions),1/self.actions)[0]]*2 # [array([1/ac,...,1/ac]),array([1/ac,...,1/ac])] 			
		else: eq = eq[0] #first nash equilibrium [[my distribution], [opponents distribution]]
		
		# print([np.full((1,self.actions),1/self.actions)])
		utils = game[eq[0],eq[1]]				
		return [eq,utils]

	
	def _invert(self,state):
		"""Returns mirror image of current state"""
		
		return (state[1],state[0])
		
	def _insert_new_state(self,state):
		"""Inserts new state and its mirror image in the memoryQ
			Assigns zero payoff matrix of proper size to both new states
		"""
		ac = self.actions
		self.memoryQ[state] = np.zeros((ac,ac))
		self.memoryQ[self._invert(state)] = np.zeros((ac,ac))
		self.memorySA[state] = np.ones((ac,ac))
		self.memorySA[self._invert(state)] = np.ones((ac,ac))
		self.memoryE[state] = 1
		self.memoryE[self._invert(state)] = 1
		
		
