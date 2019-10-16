import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Dict, Box, Discrete

class CleanupWorld(gym.Env):

    def __init__(self, max_time_steps=100):

        self.action_space_str = ['forward', 'left', 'right', 'pick']
        self.directions = {'up':0,'left':1,'down':2,'right':3}
        self.done = False
        self.map = np.zeros([8,8],dtype='uint8')
        self.goal_map = np.zeros([8,8],dtype='uint8')
        images= ['bg.png','sprite_up.png','sprite_left.png','sprite_down.png','sprite_right.png', 'obj0.png', 'obj1.png', 'obj2.png', 'obj3.png']
        self.keys = ['bg','up','left','down','right','cookie','choco','sushi','apple']
        pwd = os.getcwd()
        print(pwd)
        self.image_list = {key:cv2.imread(pwd+'/2D_cleanup/cleanup_world/envs/images/'+img) for img,key in zip(images,self.keys)}
        self.done = True
        self.agent_location = None
        self.agent_direction = None
        self.purse = None
        self.TIME_LIMIT = max_time_steps
        self.t = 0
        self.image_shape = 256
        self.observation_space = Box(high=255 * np.ones([256, 256, 3]), low=np.zeros([256, 256, 3]), dtype='uint8')
        # self.observation_space = Dict(
        #     {'goal': Box(high=255 * np.ones([256, 256, 3]), low=np.zeros([256, 256, 3]), dtype='uint8'),
        #      'observed': Box(high=255 * np.ones([256, 256, 3]), low=np.zeros([256, 256, 3]), dtype='uint8')})
        self.action_space = Discrete(4)
        self._seed()

    def _seed(self, seed=None):
        # print('set_seed', seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_location = [0, 0]
        self.agent_direction = 'up'
        positions = list(self.np_random.permutation(self.map.shape[0] * self.map.shape[1]))

        for i in range(5, 9):
            position = positions.pop()
            self.map[int(position / self.map.shape[0]), position % self.map.shape[1]] = i
        for i in range(5, 9):
            position = positions.pop()
            self.goal_map[int(position / self.map.shape[0]), position % self.map.shape[1]] = i
        # self.map[3,3] = 5 #cookie
        # self.map[7,2] = 6 #choco
        # self.map[1,1] = 7 #sushi
        # self.map[5,5] = 8 #apple
        self.map[self.agent_location[0], self.agent_location[1]] = self.directions[self.agent_direction] + 1
        self.purse = None
        self.done = False
        return self.get_obs()

    def render(self, mode='human'):
        image = np.zeros([self.map.shape[0]*32, self.map.shape[0]*32, 3],dtype='uint8')
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                # print('here')
                # print(self.image_list)
                image[i*32:(i+1)*32,j*32:(j+1)*32,:] = self.image_list[self.keys[self.map[i,j]]]

        image_goal = np.zeros([self.map.shape[0]*32, self.map.shape[0]*32, 3],dtype='uint8')
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                # print('here')
                # print(self.image_list)
                image_goal[i*32:(i+1)*32,j*32:(j+1)*32,:] = self.image_list[self.keys[self.goal_map[i,j]]]

        if mode == 'human':
            # cv2.imshow('win',np.concatenate([image,image_goal], axis=1))
            cv2.imshow('win', image)
            # cv2.waitKey(1)
        elif mode == 'rgb_array':
            return image
            # return {'goal':image_goal, 'observed':image}

    def get_neighbors(self, x, y,see_objects=True):
        up = x-1, y
        down = x+1, y
        left = x, y-1
        right = x, y+1
        neighbours = {'up': up, 'down': down, 'left': left, 'right': right}
        #         [print(v) for k,v in neighbours.items()]
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

        action = self.action_space_str[action]
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
        return obs, rew, self.done, {}

    def get_obs(self):
        return self.render(mode='rgb_array')

# if __name__ == '__main__':
#      env = CleanupWorld()
#      obs = env.reset()
#      # obs, rew, done = env.step('right')
#      # obs, rew, done = env.step('forward')
#      # obs, rew, done = env.step('right')
#      # obs, rew, done = env.step('pick')
#
#
#
#      for i in range(1000):
#          action = random.choice(env.action_space)
#          obs, rew, done = env.step(action)
#          env.difference()
#          print(rew)
#          # cv2.imwrite('env_gif/img_'+str(i).zfill(3)+'.png',obs['observed'])
#          env.render(mode='human')
#          # cv2.imshow('win', obs)
#          cv2.waitKey(100)