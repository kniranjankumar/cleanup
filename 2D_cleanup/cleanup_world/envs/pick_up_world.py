import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Dict, Box, Discrete
import copy

DIRECTIONS = {'up': 0, 'left': 1, 'down': 2, 'right': 3}


class WorldObject(object):

    def __init__(self, name, loc, direction='up'):
        assert len(loc) == 2
        self.loc = loc
        self.name = name
        self.parent = None
        self.child = None
        self.direction = direction

    def turn(self, direction):
        change = -1 if direction == 'right' else 1
        direction_num = (DIRECTIONS[self.direction]+change) % 4
        # print(direction_num)
        self.direction = [direction for direction,
                          num in DIRECTIONS.items() if num == direction_num][0]


class PickupWorld(gym.Env):

    def __init__(self, max_time_steps=100, is_goal_env=True, is_vectorized=False):
        self.world_size = 8
        self.map = np.empty((self.world_size, self.world_size), dtype=object)
        self.items = {}
        self.object_list = ['agent', 'object']
        self.add_to_world('agent', [0, 0])
        self.add_to_world('object', [5,5])
        self.TIME_LIMIT = max_time_steps
        self.action_space = Discrete(4)
        self._seed()
        self.done = True
        self.is_vectorized = is_vectorized
        if is_vectorized:
            self.observation_space = Box(high=np.ones(
                [self.world_size*self.world_size*2]), low=-1*np.ones([self.world_size*self.world_size*2]), dtype='float')
        else:
            self.observation_space = Box(high=np.ones(
                [self.world_size, self.world_size,2]), low=-1*np.ones([self.world_size, self.world_size,2]), dtype='float')

    def update_location(self, item_name, new_loc):
        item = self.items[item_name]
        self.map[item.loc[0], item.loc[1]] = None
        item.loc = new_loc
        self.map[new_loc[0], new_loc[1]] = item

    def str2objID(self,name):
        return self.object_list.index(name)+1
        
    def add_to_world(self, name, loc):
        item = WorldObject(name, loc)
        self.items[item.name] = item
        self.map[item.loc[0], item.loc[1]] = item

    def _seed(self, seed=None):
        # print('set_seed', seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_random_location(self):
        if self.positions is None:
            self.positions = list(self.np_random.permutation(
                self.map.shape[0] * self.map.shape[1]))
        position = self.positions.pop()
        return int(position / self.map.shape[0]), position % self.map.shape[1]

    def reset(self):
        # print('reset')
        self.t = 0
        self.positions = None
        self.map = np.empty((self.world_size, self.world_size), dtype=object)
        items = copy.deepcopy(self.items)
        self.items = {}
        self.done = False
        for k, v in items.items():
            loc = self.get_random_location()
            self.add_to_world(k, loc)
        return self.get_obs()

    def get_neighbors(self, loc, see_objects=True):
        x, y = loc
        up = x-1, y
        down = x+1, y
        left = x, y-1
        right = x, y+1
        neighbours = {'up': up, 'down': down, 'left': left, 'right': right}
        #         [print(v) for k,v in neighbours.items()]
        neighbours = {k: v for k, v in neighbours.items(
        ) if v[0] >= 0 and v[0] < self.map.shape[0] and v[1] >= 0 and v[1] < self.map.shape[1]}
        if see_objects:
            neighbours = {k: v for k, v in neighbours.items()
                          if self.map[v[0], v[1]] == None}
        return neighbours

        # print(self.agent_direction)

    def step(self, action):
        action_space_str = ['forward', 'left', 'right', 'pick']
        action = action_space_str[action]
        self.t += 1
        rew = -0.01
        assert self.done == False  # reset the world
        agent_location = self.items['agent'].loc
        if action == 'left':
            self.items['agent'].turn('left')
        elif action == 'right':
            self.items['agent'].turn('right')
        elif action == 'forward':
            possible_actions = self.get_neighbors(agent_location)
            # print('forward actions', possible_actions)
            if self.items['agent'].direction in possible_actions.keys():
                # print(self.agent_direction)
                move_to = possible_actions[self.items['agent'].direction]
                self.update_location('agent', move_to)
        elif action == 'pick':
            # print('here')
            possible_actions = self.get_neighbors(
                agent_location,see_objects=False)
            if self.items['agent'].direction in possible_actions.keys():
                box_in_front = possible_actions[self.items['agent'].direction]
                if isinstance(self.map[box_in_front[0], box_in_front[1]], WorldObject):
                    # pickedup object
                    rew = 1
                    self.done = True

        obs = self.get_obs()
        if self.t == self.TIME_LIMIT:
            self.done = True
        return obs, rew, self.done, {}

    @property
    def map_array(self):
        position = np.zeros_like(self.map, dtype='float')
        orientation = np.zeros_like(self.map, dtype='float')
        for k,v in self.items.items():
            position[v.loc[0],v.loc[1]] = self.str2objID(k)
            orientation[v.loc[0],v.loc[1]] = DIRECTIONS[v.direction]
        mat = np.stack([position/len(self.object_list),orientation/3],axis=2)-0.5
        return mat*2

    def get_obs(self):
        return self.map_array.reshape(-1) if self.is_vectorized else self.map_array


if __name__ == '__main__':
    env = PickupWorld(max_time_steps=1000)
    obs = env.reset()
    # obs, rew, done = env.step('right')
    # obs, rew, done = env.step('forward')
    # obs, rew, done = env.step('right')
    # obs, rew, done = env.step('pick')
    # fig, ax = plt.subplots(1, 1)
    for i in range(1000):
        #  print(i)
        action = env.action_space.sample()
        obs, rew, done, _ = env.step(action)
        plt.matshow(np.sum(obs,2),0,vmax = 1)
        plt.pause(0.05)
        # plt.show()
        # env.render()
        if done:
            env.reset()
        # env.difference()
        # print(rew)
        # cv2.imwrite('env_gif/img_'+str(i).zfill(3)+'.png',obs['observed'])
        #  env.render(mode='human')
        # cv2.imshow('win', obs)
        cv2.waitKey(10)
    plt.show()