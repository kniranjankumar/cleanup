## Generic Python Modules
import pickle
import numpy as np
from copy import deepcopy

## TO DO: Need to change this to suit the new coordinate system
def get_location_orientation(observation_matrix, no):
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

def return_dataset(data):
	## construct the dataset
	X = []
	Y = []

	## to store the observations till a desired pick action
	X_new = []

	for value in data:
		if value['action'] != 'pick':
			X_new.append(value['obs'].flatten())
			# print(value['obs'][:,:,0])
			# print(value['action'])
		else:
			XX = np.array(X_new)
			X.append(deepcopy(XX))
			XXX = np.array(X)
			# print(XX.shape)
			robot_loc, robot_orient = get_location_orientation(deepcopy(value['obs'][:,:,0]),1.0)
			output_matrix = np.zeros((8,8))
			output_matrix[robot_loc[0]][robot_loc[1]] = 1
			output_matrix = output_matrix.flatten()
			# print(output_matrix)
			output_matrix = np.append(output_matrix, robot_orient)
			Y.append(output_matrix)
			# print(output_matrix)
			# print(robot_loc)
			# print(value['obs'][:,:,0])
			# print(value['action'])
			X_new = []

	X = np.array(X)
	Y = np.array(Y)

	return X, Y 

if __name__=="__main__":
	
	## load the data from pickle file
	## need to do this for 6 episodes
	with open('/home/nithin/Desktop/cleanup/new_cleanup/cleanup/collected_data/subject_1/eps_1.pkl','rb') as f:
		data = pickle.load(f)

	X, Y = return_dataset(deepcopy(data))
	print(X.shape)
	print(Y.shape)