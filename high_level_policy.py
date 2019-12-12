## Generic Python Modules
import pickle
import numpy as np
from copy import deepcopy
from trajectory_extraction import *
import random
from os import walk, listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

## Deep Learning Modules
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from keras import optimizers
from keras import regularizers
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
from tqdm import tqdm
from keras_tqdm import TQDMCallback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import Normalizer, StandardScaler

## Fix Random Seed
np.random.seed(11)
tf.set_random_seed(11)

def generate_random_sample(X):
	X_random = []
	for x in X:
		idx = np.random.randint(x.shape[0])
		# X_random.append(x[idx])
		X_random.append(x[0])
	X_random = np.array(X_random)
	return X_random


if __name__=="__main__":
	
	## actions
	forward = [1,0,0,0]
	left = [0,1,0,0]
	right = [0,0,1,0]
	pick = [0,0,0,1]

	X_total = []
	Y1_total = []
	Y2_total = []

	## load the data from pickle file
	folder = '/home/nithin/Desktop/cleanup/clean_data/'
	all_files = []
	for (dirpath, _, _) in walk(folder):
		onlyfiles = [join(dirpath, f) for f in listdir(dirpath) if isfile(join(dirpath, f))]
		if len(onlyfiles) > 1:
			all_files.append(onlyfiles)

	for file in all_files:
		for infile in file:
			# print(infile)
			with open(infile,'rb') as d:
				data = pickle.load(d)
				X, Y1, Y2 = return_dataset(deepcopy(data))
				X = generate_random_sample(deepcopy(X))
				for x,y1,y2 in zip(X,Y1,Y2):
					X_total.append(x)
					Y1_total.append(y1)
					Y2_total.append(y2)

	## construct the dataset
	X_total = np.array(X_total)
	Y1_total = np.array(Y1_total)
	Y2_total = np.array(Y2_total)

	print(X_total.shape)
	print(Y1_total.shape)
	print(Y2_total.shape)

	X_total_new = []
	for x in X_total:
		x = (x - np.mean(x))/np.std(x)
		X_total_new.append(x)

	X_total_new = np.array(X_total_new)
	X_total = deepcopy(X_total_new)


	#### FOR CONV POLICY
	# input_shape = [8,8,2]
	# n_classes1 = 64
	# n_classes2 = 4


	# # ## split into training and testing set
	# X_total, Y1_total, Y2_total = shuffle(X_total,Y1_total,Y2_total)
	# X_train, X_test, Y_train1, Y_test1, Y_train2, Y_test2 = train_test_split(X_total, Y1_total, Y2_total, test_size=0.2)

	# ## build model
	# inputs = Input(input_shape)
	# x = Conv2D(8, (2,2), padding="same", strides=(1, 1), activation='relu', name='Layer1')(inputs)
	# x = MaxPooling2D(pool_size=(2, 2))(x)
	# # x = Dropout(0.5)(x)
	# x = Conv2D(16, (2,2), padding="same", strides=(1,1), activation='relu', name='Layer2')(x)
	# x = MaxPooling2D(pool_size=(2, 2))(x)
	# # x = Dropout(0.5)(x)
	# x = Flatten()(x)
	# prediction_location = Dense(n_classes1, activation='softmax')(x)
	# prediction_orientation = Dense(n_classes2, activation='softmax')(x)

	# model = Model(inputs = inputs, outputs = [prediction_location, prediction_orientation])
	# # model = Model(inputs = inputs, outputs = prediction_location)

	# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001), metrics=['accuracy'])
	# # history = model.fit(X_train, [Y_train1, Y_train2], validation_split=0.2, epochs=10, batch_size=25, verbose=0, callbacks=[TQDMCallback()])
	# history = model.fit(X_train, [Y_train1,Y_train2], validation_split=0.2, epochs=1000, batch_size=5, verbose=0, callbacks=[TQDMCallback()])

	# ## model plots
	# print(history.history.keys())
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()
	

	# #### FOR MLP POLICY
	input_shape = 128
	n_classes1 = 64
	n_classes2 = 4


	## split into training and testing set
	X_total, Y1_total, Y2_total = shuffle(X_total,Y1_total,Y2_total)
	scaler = StandardScaler()
	X_total = scaler.fit_transform(X_total)
	X_train, X_test, Y_train1, Y_test1, Y_train2, Y_test2 = train_test_split(X_total, Y1_total, Y2_total, test_size=0.2)

	## build model
	inputs = Input([input_shape])
	x = Dense(85, activation='relu', name='Layer1')(inputs)
	# x = Dropout(0.5)(x)
	x = Dense(50, activation='relu', name='Layer2')(x)
	# x = Dropout(0.5)(x)
	x = Dense(25, activation='relu', name='Layer3')(x)
	prediction_location = Dense(n_classes1, activation='softmax')(x)
	prediction_orientation = Dense(n_classes2, activation='softmax')(x)

	model = Model(inputs = inputs, outputs = [prediction_location, prediction_orientation])
	# model = Model(inputs = inputs, outputs = prediction_location)

	model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
	# history = model.fit(X_train, [Y_train1, Y_train2], validation_split=0.2, epochs=10, batch_size=25, verbose=0, callbacks=[TQDMCallback()])
	history = model.fit(X_train, [Y_train1,Y_train2], validation_split=0.2, epochs=1000, batch_size=5, verbose=0, callbacks=[TQDMCallback()])

	## model plots
	print(history.history.keys())
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()
