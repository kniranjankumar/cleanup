import gym
import numpy as np
from gym.envs.registration import register
from matplotlib import pyplot as plt
import time
from astar import *
from copy import deepcopy

def return_direction(tuple1, tuple2):
	if ((tuple1[0]-tuple2[0])==1) and ((tuple1[1]-tuple2[1])==0):
		return 0
	elif ((tuple1[0]-tuple2[0])==-1) and ((tuple1[1]-tuple2[1])==0):
		return 1
	elif ((tuple1[0]-tuple2[0])==0) and ((tuple1[1]-tuple2[1])==-1):
		return 2
	elif ((tuple1[0]-tuple2[0])==0) and ((tuple1[1]-tuple2[1])==1):
		return 3

def get_location_orientation(observation_matrix):
	robot_location_1 = np.where(observation_matrix==1)
	robot_location_2 = np.where(observation_matrix==1.25)
	robot_location_3 = np.where(observation_matrix==1.5)
	robot_location_4 = np.where(observation_matrix==1.75)

	if len(robot_location_1[0])!=0:
		return tuple([robot_location_1[0][0], robot_location_1[1][0]]), 0
	elif len(robot_location_2[0])!=0:
		return tuple([robot_location_2[0][0], robot_location_2[1][0]]), 3
	elif len(robot_location_3[0])!=0:
		return tuple([robot_location_3[0][0], robot_location_3[1][0]]), 1
	elif len(robot_location_4[0])!=0:
		return tuple([robot_location_4[0][0], robot_location_4[1][0]]), 2


if __name__=="__main__":
	## transition matrix
	right = np.array([2,0])
	left = np.array([1,0])
	right_right = np.array([2,2,0])
	T = np.array(([0, [2, 2, 0], [2, 0], [1, 0]],[[2, 2, 0], 0, [1, 0], [2, 0]],[[1, 0], [2, 0], 0, [2, 2, 0]],[[2, 0], [1, 0], [2, 2, 0], 0]))

	## create the gym environment
	register(id='2DPickup-v2', entry_point='cleanup_world.envs:PickupWorld', kwargs={'render':True})
	env = gym.make('2DPickup-v2')
	obs = env.reset()

	## astar test 
	# achieved_goal = obs['achieved_goal']
	# observation = obs['observation']
	# desired_goal = obs['desired_goal']

	# desired_goal = desired_goal.reshape((8,8,2))
	# desired_goal_matrix = desired_goal[:,:,1]
	# print(desired_goal_matrix)
	# observation = observation.reshape((8,8,2))
	# observation_matrix = observation[:,:,0]
	# maze = deepcopy(observation_matrix)
	# print(observation_matrix)
	# start = np.where(observation_matrix==1)
	# start, orientation = get_location_orientation(deepcopy(observation_matrix))
	# end = (0,0)
	# path = astar(maze, start, end)
	# print(path)


	eps = 1
	done = False
	rew_list = []
	print("hi")
	for i in range(eps):

		## get the observation
		obs_temp = obs['observation']
		obs_temp = obs_temp.reshape((8,8,2))
		maze = deepcopy(obs_temp[:,:,0])
		observation_matrix = deepcopy(maze)
		print(observation_matrix)
		
		## astar
		# robot_location = np.where(observation_matrix==1)
		# chair_location = np.where(observation_matrix==2)
		start, orientation = get_location_orientation(deepcopy(observation_matrix))
		# start = tuple([robot_location[0][0], robot_location[1][0]])
		end = (0,0)
		path = astar(maze, start, end)
		path.remove(start)

		for state in path:
			start, orientation = get_location_orientation(deepcopy(observation_matrix))
			print("orientation: ", orientation)
			tuple2 = deepcopy(state)
			tuple1 = deepcopy(start)
			direction = return_direction(tuple1, tuple2)
			print("direction: ", direction)
			actions_to_perform = T[orientation][direction]
			print("actions_to_perform: ", actions_to_perform)
			print("isinstance: ", isinstance(actions_to_perform, list))
			if isinstance(actions_to_perform, list):
				for action1 in actions_to_perform:
					obs,rew,done,_ = env.step(action1)
					rew_list.append(rew)
					obs_temp = obs['observation']
					obs_temp = obs_temp.reshape((8,8,2))
					observation_matrix = obs_temp[:,:,0]
			else:
				obs,rew,done,_ = env.step(actions_to_perform)
				rew_list.append(rew)
				obs_temp = obs['observation']
				obs_temp = obs_temp.reshape((8,8,2))
				observation_matrix = obs_temp[:,:,0]
				print(observation_matrix)

		
		print('done')
		env.reset()
