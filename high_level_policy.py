## Generic Python Modules
import pickle
import numpy as np
from copy import deepcopy

## Deep Learning Modules
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

## Fix Random Seed
np.random.seed(11)

def onehot(output):
	## actions
	forward = [1,0,0,0]
	left = [0,1,0,0]
	right = [0,0,1,0]
	pick = [0,0,0,1]

	result = []
	## convert to one hot vector
	max_indices = np.argmax(output, axis=1)
	for index in max_indices:
		if index == 0:
			result.append(forward)
		elif index == 1:
			result.append(left)
		elif index == 2:
			result.append(right)
		elif index == 3:
			result.append(pick)

	result = np.array(result)

	return result

if __name__=="__main__":
	
	## actions
	forward = [1,0,0,0]
	left = [0,1,0,0]
	right = [0,0,1,0]
	pick = [0,0,0,1]

	## load the data from pickle file
	## need to do this for 6 episodes
	with open('/home/nithin/Desktop/cleanup/collected_data/eps_1.pkl','rb') as f:
		data = pickle.load(f)

	## construct the dataset
	X = []
	Y = []
	for value in data:
		X.append(value['obs'].flatten())
		if value['action'] == 'forward':
			Y.append(forward)
		elif value['action'] == 'left':
			Y.append(left)
		elif value['action'] == 'right':
			Y.append(right)
		elif value['action'] == 'pick':
			Y.append(pick)

	X = np.array(X)
	Y = np.array(Y)

	input_shape = X.shape[1:]
	n_classes = 4

	## split into training and testing set
	X,Y = shuffle(X,Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

	## build model
	inputs = Input(input_shape)
	x = Dense(85, activation='relu', name='Layer1')(inputs)
	x = Dropout(0.15)(x)
	x = Dense(50, activation='relu', name='Layer2')(x)
	x = Dropout(0.05)(x)
	x = Dense(25, activation='relu', name='Layer3')(x)
	prediction = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs = inputs, outputs = prediction)

	## compile the model
	model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
	model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=25)
	results = model.predict(X_train)
	Y_predict = onehot(deepcopy(results))
	# print(Y_predict)
	# print(Y_train)
	# print(Y_train==Y_predict)










