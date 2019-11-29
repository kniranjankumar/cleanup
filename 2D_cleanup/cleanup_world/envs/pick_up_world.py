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

DIRECTIONS = {"up": 0, "left": 1, "down": 2, "right": 3}


class WorldObject(object):
    def __init__(self, name, loc, map, direction="up"):
        assert len(loc) == 2
        self.loc = loc
        self.map = map
        self.name = name
        self.parent = None
        self.child = None
        self.direction = direction
        self.goal_loc = loc
        self.goal_direction = direction
        self.map = map

    def turn(self, direction):
        change = -1 if direction == "right" else 1
        direction_num = (DIRECTIONS[self.direction] + change) % 4
        # print(direction_num)
        self.direction = [
            direction for direction, num in DIRECTIONS.items() if num == direction_num
        ][0]

    @property
    def neighbors(self):
        x, y = self.loc
        up = x - 1, y
        down = x + 1, y
        left = x, y - 1
        right = x, y + 1
        neighbors = {"up": up, "down": down, "left": left, "right": right}
        #         [print(v) for k,v in neighbours.items()]
        neighbors = {
            k: v
            for k, v in neighbors.items()
            if v[0] >= 0
            and v[0] < self.map.shape[0]
            and v[1] >= 0
            and v[1] < self.map.shape[1]
        }
        return neighbors

    @property
    def empty_neighbors(self):
        empty_neighbors = {
            k: v for k, v in self.neighbors.items() if self.map[v[0], v[1]] == None
        }
        return empty_neighbors

    @property
    def box_in_front(self):
        if self.direction in self.neighbors.keys():
            box_in_front = self.neighbors[self.direction]
        else:
            box_in_front = None
        return box_in_front


class PickupWorld(gym.Env):
    def __init__(
        self,
        max_time_steps=100,
        is_goal_env=False,
        is_vectorized=False,
        is_random_start=True,
    ):
        self.world_size = 8
        self.map = np.empty((self.world_size, self.world_size), dtype=object)
        self.items = {}
        self.object_list = ["agent", "object"]
        self.add_to_world("agent", [0, 0])
        self.add_to_world("object", [5, 5])
        self.TIME_LIMIT = max_time_steps
        self.action_space = Discrete(4)
        self._seed()
        self.done = True
        self.is_vectorized = is_vectorized
        self.is_random_start = is_random_start
        self.is_goal_env = is_goal_env
        vector_box = Box(
            high=np.ones([self.world_size * self.world_size * 2]),
            low=-1 * np.ones([self.world_size * self.world_size * 2]),
            dtype="float",
        )
        grid_box = Box(
            high=np.ones([self.world_size, self.world_size, 2]),
            low=-1 * np.ones([self.world_size, self.world_size, 2]),
            dtype="float",
        )
        if self.is_goal_env:
            self.observation_space = Dict(
                {
                    "observation": vector_box if self.is_vectorized else grid_box,
                    "achieved_goal": vector_box if self.is_vectorized else grid_box,
                    "desired_goal": vector_box if self.is_vectorized else grid_box,
                }
            )
        else:
            self.observation_space = vector_box if self.is_vectorized else grid_box

    def update_location(self, item_name, new_loc):
        item = self.items[item_name]
        self.map[item.loc[0], item.loc[1]] = None
        item.loc = new_loc
        self.map[new_loc[0], new_loc[1]] = item

    def str2objID(self, name):
        return self.object_list.index(name) + 1

    def add_to_world(self, name, loc):
        item = WorldObject(name, loc, self.map)
        self.items[item.name] = item
        self.map[item.loc[0], item.loc[1]] = item

    def _seed(self, seed=None):
        # print('set_seed', seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_random_location(self, obj_name):
        if self.positions is None:
            self.positions = list(
                self.np_random.permutation(
                    np.arange(1, self.map.shape[0] * self.map.shape[1])
                )
            )
        if obj_name == "agent":
            position = 0
        else:
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
            loc = self.get_random_location(k)
            self.add_to_world(k, loc)
        self.create_goal()
        return self.get_obs()

    def get_neighbors(self, loc, see_objects=True):
        x, y = loc
        up = x - 1, y
        down = x + 1, y
        left = x, y - 1
        right = x, y + 1
        neighbours = {"up": up, "down": down, "left": left, "right": right}
        #         [print(v) for k,v in neighbours.items()]
        neighbours = {
            k: v
            for k, v in neighbours.items()
            if v[0] >= 0
            and v[0] < self.map.shape[0]
            and v[1] >= 0
            and v[1] < self.map.shape[1]
        }
        if see_objects:
            neighbours = {
                k: v for k, v in neighbours.items() if self.map[v[0], v[1]] == None
            }
        return neighbours

        # print(self.agent_direction)

    def step(self, action):
        action_space_str = ["forward", "left", "right", "pass"]
        action = action_space_str[action]
        self.t += 1
        rew = -0.0
        assert self.done == False  # reset the world
        agent_location = self.items["agent"].loc
        box_in_front = self.items["agent"].box_in_front
        is_empty = False
        if box_in_front != None:
            is_empty = not isinstance(
                    self.map[box_in_front[0], box_in_front[1]], WorldObject
                )  # is grid empty?
        if action == "left":
            self.items["agent"].turn("left")
        elif action == "right":
            self.items["agent"].turn("right")
        elif action == "forward":
            if is_empty:
                # print(self.agent_direction)
                self.update_location("agent", box_in_front)
        elif action == "pass":
            pass
            # self.done = True

            # possible_actions = self.get_neighbors(
            #     agent_location,see_objects=False)
            # if self.items['agent'].direction in possible_actions.keys():
            #     box_in_front = possible_actions[self.items['agent'].direction]
            #     if isinstance(self.map[box_in_front[0], box_in_front[1]], WorldObject):
            #         # pickedup object
            #         rew = 1
            # # possible_actions = self.get_neighbors(
            # #     agent_location,see_objects=False)
            # # if self.items['agent'].direction in possible_actions.keys():
            # #     box_in_front = possible_actions[self.items['agent'].direction]
            # #     if isinstance(self.map[box_in_front[0], box_in_front[1]], WorldObject):
            # #         # pickedup object
            # #         rew = 1
            # #         self.done = True
        if not is_empty:
            rew = 1
        obs = self.get_obs()
        if self.t == self.TIME_LIMIT:
            self.done = True
        return obs, rew, self.done, {}

    def normalize_array(self, array):
        return (array - 0.5) * 2

    def unnormalize_array(self, array):
        return array / 2 + 0.5

    @property
    def map_array(self):
        position = np.zeros_like(self.map, dtype="float")
        orientation = np.zeros_like(self.map, dtype="float")
        for k, v in self.items.items():
            position[v.loc[0], v.loc[1]] = self.str2objID(k)
            orientation[v.loc[0], v.loc[1]] = DIRECTIONS[v.direction]
        mat = np.stack([position / len(self.object_list), orientation / 3], axis=2)
        return self.normalize_array(mat)

    @property
    def goal_array(self):
        position = np.zeros_like(self.map, dtype="float")
        orientation = np.zeros_like(self.map, dtype="float")
        position[self.items['object'].goal_loc[0], self.items['object'].goal_loc[1]] = self.str2objID('object')
        orientation[self.items['object'].goal_loc[0], self.items['object'].goal_loc[1]] = DIRECTIONS[self.items['object'].goal_direction]
        # for k, v in self.items.items():
        #     position[v.goal_loc[0], v.goal_loc[1]] = self.str2objID(k)
        #     orientation[v.goal_loc[0], v.goal_loc[1]] = DIRECTIONS[v.goal_direction]
        mat = np.stack([position / len(self.object_list), orientation / 3], axis=2)
        return self.normalize_array(mat)
        # return self.self.items['object'].goal_loc

    @property
    def achieved_array(self):
        position = np.zeros_like(self.map, dtype="float")
        orientation = np.zeros_like(self.map, dtype="float")
        # agent location
        # position[
        #     self.items["agent"].loc[0], self.items["agent"].loc[1]
        # ] = self.str2objID("agent")
        # orientation[
        #     self.items["agent"].loc[0], self.items["agent"].loc[1]
        # ] = DIRECTIONS[self.items["agent"].direction]
        # object location
        if self.items["agent"].box_in_front is not None:
            print("here")
            position[
                self.items["agent"].box_in_front[0], self.items["agent"].box_in_front[1]
            ] = self.str2objID("object")
            orientation[
                self.items["agent"].box_in_front[0], self.items["agent"].box_in_front[1]
            ] = DIRECTIONS[self.items["object"].goal_direction]
        mat = np.stack([position / len(self.object_list), orientation / 3], axis=2)
        return self.normalize_array(mat)

    def create_goal(self):
        self.goal_map = np.empty((self.world_size, self.world_size), dtype=object)
        neighbors = self.items["object"].neighbors
        # neighbors = self.get_neighbors(self, self.items['object'].loc, see_objects=True)
        # choose a neighbor cell and place agent there
        selected_cell_direction = self.np_random.choice(list(neighbors.keys()))
        agent_location = neighbors[selected_cell_direction]
        inverse_directions = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }
        print(selected_cell_direction)
        agent_direction = inverse_directions[selected_cell_direction]
        self.items["agent"].goal_loc = agent_location
        self.items["agent"].goal_direction = agent_direction

    def get_obs(self):
        if not self.is_goal_env:
            return self.map_array.reshape(-1) if self.is_vectorized else self.map_array
        else:
            return {
                "observation": self.map_array.reshape(-1)
                if self.is_vectorized
                else self.map_array,
                "achieved_goal": self.achieved_array.reshape(-1)
                if self.is_vectorized
                else self.achieved_array.reshape(-1),
                "desired_goal": self.goal_array.reshape(-1)
                if self.is_vectorized
                else self.goal_array.reshape(-1),
            }

    def get_position_from_array(self, array, objectID):
        position = array[:,:,0]*len(self.object_list) 
        location = np.argwhere(position == objectID)
        return location

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = achieved_goal.reshape(self.map.shape)
        desired_goal = desired_goal.reshape(self.map.shape)
        achieved_goal = self.unnormalize_array(achieved_goal)
        desired_goal = self.unnormalize_array(desired_goal)
        achieved_loc = self.get_position_from_array(achieved_goal,self.str2objID('object'))
        desired_loc = self.get_position_from_array(desired_goal,self.str2objID('object'))
        if len(achieved_loc)==0:
            x,y = self.items["agent"].loc
            unbound_neighbors = {'up':[x - 1, y],'down':[x + 1, y],'left':[x, y - 1],'right':[x, y + 1]}
            achieved_loc = unbound_neighbors[self.items['agent'].direction]
        print(achieved_loc,desired_loc)
        distance = np.linalg.norm(desired_loc[0]-achieved_loc[0],1)
        # return distance
        return 2-2*distance/(self.map.shape[0]*self.map.shape[1])

if __name__ == "__main__":
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
        plt.matshow(np.sum(obs, 2), 0, vmax=1)
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
