## generic python packages
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys

## python packages for pygame
import pygame as pg
from pygame.locals import *

## python packages for gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Dict, Box, Discrete

## Class to represent a single grid in the grid world
class Grid:
	def __init__(self, x, y, image, game_window):
		self.grid_location = (x, y)
		self.grid_coordinates = [x//60, y//60]
		self.image = pg.image.load(image)
		self.game_window = game_window

	def draw(self):
		grid = pg.Rect(self.grid_location, (60,60))
		self.image = pg.transform.scale(self.image, (60, 60))
		pg.draw.rect(self.game_window, pg.Color('white'), grid, 0)
		self.game_window.blit(self.image, self.grid_location)
		pg.draw.rect(self.game_window, pg.Color('blue'), grid, 2)

## main class which has the world
class CleanupWorld(gym.Env):
	def __init__(self, game_window, max_time_steps=100):

		## random number for random locations of objects
		random_number_1 = random.randint(0,3)
		random_number_2 = random.randint(0,3)
		random_number_3 = random.randint(0,1)
		random_number_4 = random.randint(0,1)

		## parameters for pygame
		self.grid_width = 60
		self.grid_height = 60
		self.game_window = game_window 

		## images for the random objects
		self.chair_image = ['chair_down.png', 'chair_left.png', 'chair_right.png', 'chair_up.png']
		self.couch_image = ['couch_down.png', 'couch_left.png', 'couch_right.png', 'couch_up.png']
		self.cupboard_image = ['cupboard_front.png', 'cupboard_rear.png']
		self.table_image = ['table_front.png', 'table_rear.png']

		## parameters for the world
		self.action_space_str = ['forward', 'left', 'right', 'pick']
		self.map_action_to_index = {'forward':0, 'left':1, 'right':2, 'pick':3}
		self.directions = {'up':0,'left':1,'down':2,'right':3}
		self.done = False  
		self.map = np.zeros([16,16],dtype='uint8')
		self.goal_map = np.zeros([16,16],dtype='uint8')
		self.images= ['bg.png','sprite_up.png','sprite_left.png','sprite_down.png','sprite_right.png', 'obj0.png', 'obj1.png', 'obj2.png', 'obj3.png', self.chair_image[random_number_1], 'coffee_cup.png', self.couch_image[random_number_2], self.cupboard_image[random_number_3], 'laptop.png', 'phone.png', 'plates.png', self.table_image[random_number_4], 'tea_table.png']
		self.keys = ['bg','up','left','down','right','cookie','choco','sushi','apple', 'chair', 'coffee_cup', 'couch', 'cupboard', 'laptop', 'phone', 'plates', 'table', 'tea_table']
		self.image_list = {key:'/home/nithin/Desktop/cleanup/cleanup/2D_cleanup/cleanup_world/envs/images/scene_objects/'+img for img,key in zip(self.images,self.keys)}
		self.done = True
		self.agent_location = None
		self.agent_direction = None
		self.purse = None
		self.TIME_LIMIT = max_time_steps
		self.t = 0
		self.observation_space = Dict(
		    {'goal': Box(high=255 * np.ones([256, 256, 3]), low=np.zeros([256, 256, 3]), dtype='uint8'),
		     'observed': Box(high=255 * np.ones([256, 256, 3]), low=np.zeros([256, 256, 3]), dtype='uint8')})
		self.action_space = Discrete(4)
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.agent_location = [0, 0]
		self.agent_direction = 'up'
		positions = list(self.np_random.permutation(self.map.shape[0] * self.map.shape[1]))

		for i in range(5, 18):
			position = positions.pop()
			self.map[int(position / self.map.shape[0]), position % self.map.shape[1]] = i

		for i in range(5, 18):
			position = positions.pop()
			self.goal_map[int(position / self.map.shape[0]), position % self.map.shape[1]] = i
		
		self.map[self.agent_location[0], self.agent_location[1]] = self.directions[self.agent_direction] + 1
		self.purse = None
		self.done = False
		
		return self.get_obs()

	def render(self, mode='human'):
		self.board = []
		self.board_size = [self.map.shape[0], self.map.shape[1]]
		for i in range(0, 16):
			row = []
			for j in range(0, 16):
				grid_index = i * 16 + j
				x = j * self.grid_width
				y = i * self.grid_height
				grid = Grid(x, y, self.image_list[self.keys[self.map[i,j]]], self.game_window)
				row.append(grid)
			self.board.append(row)

		for each_row in self.board:
			for each_grid in each_row:
				each_grid.draw()

	def get_neighbors(self, x, y,see_objects=True):
		up = x-1, y
		down = x+1, y
		left = x, y-1
		right = x, y+1
		neighbours = {'up': up, 'down': down, 'left': left, 'right': right}
		neighbours = {k: v for k, v in neighbours.items() if v[0] >= 0 and v[0] < 4 and v[1] >= 0 and v[1] < 4}
		if see_objects:
			neighbours = {k: v for k, v in neighbours.items() if self.map[v[0],v[1]] == 0}
		return neighbours

	def turn(self, direction):
		change = -1 if direction == 'right' else 1
		direction_num = (self.directions[self.agent_direction]+change)%4
		# print(direction_num)
		self.agent_direction = [direction for direction, num in self.directions.items() if num == direction_num][0]
		self.map[self.agent_location[0], self.agent_location[1]] = self.directions[self.agent_direction] + 1
		# print(self.agent_direction)

	def difference(self):
		temp = np.copy(self.map)
		temp[self.agent_location[0], self.agent_location[1]] = 0
		diff = self.goal_map!=temp
		# diff[self.agent_location[0], self.agent_location[1]] = False
		return np.sum(diff.astype('uint8'))

	def step(self, action):
		action = self.action_space_str[self.map_action_to_index[action]]
		self.t += 1
		assert self.done == False  # reset the world
		if action == 'left':
			self.turn('left')
		elif action == 'right':
			self.turn('right')
		elif action == 'forward':
			possible_actions = self.get_neighbors(self.agent_location[0], self.agent_location[1])
			# print('forward actions', possible_actions)
			if self.agent_direction in possible_actions.keys():
				# print(self.agent_direction)
				move_to = possible_actions[self.agent_direction]
				self.map[self.agent_location[0], self.agent_location[1]] = 0
				self.agent_location = move_to
				self.map[self.agent_location[0], self.agent_location[1]] = self.directions[self.agent_direction] + 1
		elif action == 'pick':
			if self.purse is None:
				possible_actions = self.get_neighbors(self.agent_location[0], self.agent_location[1],see_objects=False)
				if self.agent_direction in possible_actions.keys():
					# print('here')
					box_in_front = possible_actions[self.agent_direction]
					obj_in_front = self.map[box_in_front[0], box_in_front[1]]
					# print(obj_in_front)
					if 5 <= obj_in_front <= 8:
						# print(obj_in_front)
						self.purse = obj_in_front
						self.map[box_in_front[0], box_in_front[1]] = 0
			else:
				possible_actions = self.get_neighbors(self.agent_location[0], self.agent_location[1])
				if self.agent_direction in possible_actions.keys():
					box_in_front = possible_actions[self.agent_direction]
					self.map[box_in_front[0], box_in_front[1]] = self.purse
					self.purse = None

		obs = self.get_obs()
		diff = self.difference()
		rew = 1-diff/4
		if rew == 1:
			self.done = True
		elif self.t == self.TIME_LIMIT:
			self.done = True
		return obs, rew, self.done

	def get_obs(self):
		return self.render(mode='rgb_array')
