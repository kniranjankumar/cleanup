import gym
import numpy as np
from gym.envs.registration import register
from matplotlib import pyplot as plt
import time
from astar import *
from copy import deepcopy
import random

class OraclePlanner(object):

	def __init__(self, obs):
			## transition matrix
			self.T = np.array(([0, [2, 2, 0], [2, 0], [1, 0]],[[2, 2, 0], 0, [1, 0], [2, 0]],[[1, 0], [2, 0], 0, [2, 2, 0]],[[2, 0], [1, 0], [2, 2, 0], 0]))

			## get observation matrix
			obs_temp = obs['observation']
			obs_temp = obs_temp.reshape((8,8,2))
			self.observation_matrix = obs_temp[:,:,0]
			self.maze = deepcopy(self.observation_matrix)

			## get the desired matrix
			desired_temp = obs['desired_goal']
			desired_temp = desired_temp.reshape((8,8,2))
			self.desired_goal_matrix = desired_temp[:,:,1]

			## get the start pose of robot
			self.start, self.orientation = self.get_location_orientation(deepcopy(self.observation_matrix), 1.0)

			## get the chair pose	
			self.end, self.chair_orientation = self.get_location_orientation(deepcopy(self.desired_goal_matrix), 1.0)
			self.end_goal = self.get_neighbors(deepcopy(self.observation_matrix), self.end)

			## get the path
			self.path = astar(self.maze, self.start, self.end_goal)
			self.path.remove(self.start)


	def return_direction(self, tuple1, tuple2):
		if ((tuple1[0]-tuple2[0])==1) and ((tuple1[1]-tuple2[1])==0):
			return 0
		elif ((tuple1[0]-tuple2[0])==-1) and ((tuple1[1]-tuple2[1])==0):
			return 1
		elif ((tuple1[0]-tuple2[0])==0) and ((tuple1[1]-tuple2[1])==-1):
			return 2
		elif ((tuple1[0]-tuple2[0])==0) and ((tuple1[1]-tuple2[1])==1):
			return 3

	def get_location_orientation(self, observation_matrix, no):
		robot_location_1 = np.where(observation_matrix==no)
		robot_location_2 = np.where(observation_matrix==(no+0.25))
		robot_location_3 = np.where(observation_matrix==(no+0.50))
		robot_location_4 = np.where(observation_matrix==(no+0.75))

		if len(robot_location_1[0])!=0:
			return tuple([robot_location_1[0][0], robot_location_1[1][0]]), 0
		elif len(robot_location_2[0])!=0:
			return tuple([robot_location_2[0][0], robot_location_2[1][0]]), 3
		elif len(robot_location_3[0])!=0:
			return tuple([robot_location_3[0][0], robot_location_3[1][0]]), 1
		elif len(robot_location_4[0])!=0:
			return tuple([robot_location_4[0][0], robot_location_4[1][0]]), 2

	def get_neighbors(self, observation_matrix, location):
		neighbors = []
		for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
			# Get node position
			node_position = (location[0] + new_position[0], location[1] + new_position[1])

			# Make sure within range
			if node_position[0] > (len(observation_matrix) - 1) or node_position[0] < 0 or node_position[1] > (len(observation_matrix[len(observation_matrix)-1]) -1) or node_position[1] < 0:
				continue

			# Make sure walkable terrain
			if observation_matrix[node_position[0]][node_position[1]] != 0:
				continue
			neighbors.append(node_position)
		return random.choice(neighbors)

	def oracle_step(self, obs):

		if(len(self.path)>0):
			## get observation matrix
			obs_temp = obs['observation']
			obs_temp = obs_temp.reshape((8,8,2))
			self.observation_matrix = obs_temp[:,:,0]
			
			## get the current state
			self.start, self.orientation = self.get_location_orientation(deepcopy(self.observation_matrix), 1.0)
			tuple2 = deepcopy(self.path[0])
			tuple1 = deepcopy(self.start)
			self.direction = self.return_direction(tuple1, tuple2)

			## get the actions to perform
			actions_to_perform = self.T[self.orientation][self.direction]

			## pop the state
			self.path.pop(0)
			return actions_to_perform
		
		else:
			return None 


if __name__=="__main__":
	## create the gym environment
	register(id='2DPickup-v2', entry_point='cleanup_world.envs:PickupWorld', kwargs={'render':True})
	env = gym.make('2DPickup-v2')
	obs = env.reset()

	eps = 1
	done = False
	rew_list = []
	for i in range(eps):
		oracle_object = OraclePlanner(obs)

		while not done:
		
			actions_to_perform = oracle_object.oracle_step(obs)
			print(actions_to_perform)

			if actions_to_perform is not None:
				if isinstance(actions_to_perform, list):
						for action1 in actions_to_perform:
							obs,rew,done,_ = env.step(action1)
							rew_list.append(rew)
							# obs_temp = obs['observation']
							# obs_temp = obs_temp.reshape((8,8,2))
							# observation_matrix = obs_temp[:,:,0]
							# print(observation_matrix)
				else:
					obs,rew,done,_ = env.step(actions_to_perform)
					rew_list.append(rew)
					# obs_temp = obs['observation']
					# obs_temp = obs_temp.reshape((8,8,2))
					# observation_matrix = obs_temp[:,:,0]
					# print(observation_matrix)
			
			else:
				obs,rew,done,_ = env.step(4)

		
		print('done')
		# env.reset()
