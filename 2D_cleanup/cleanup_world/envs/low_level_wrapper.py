from cleanup_world.envs.clean_up_world import CleanupWorld
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from gym.spaces import Dict, Box, Discrete

import pickle
class PickupWorld(CleanupWorld):
    def __init__(self, max_time_steps=100, is_goal_env=True, render=False):
        super().__init__(max_time_steps=max_time_steps,
                         render=render)
        self.action_space = Discrete(4)
        vector_box = Box(
            high=np.ones([self.world_size * self.world_size]),
            low=-1 * np.ones([self.world_size * self.world_size]),
            dtype="float",
        )
        self.is_goal_env = is_goal_env
        if is_goal_env:
            with open('./clean_data/all_trajectories.pkl','rb') as f:
                data = pickle.load(f)
                self.expert_state = data['observations']
                self.expert_goals = data['goals']
                self.expert_goal_orientations = data['orientation']
            self.observation_space = Dict(
                {
                    "observation": vector_box,
                    "achieved_goal": vector_box,
                    "desired_goal": vector_box,
                }
            )
        else:
            self.observation_space = vector_box
            
    def reset(self):
        obs = super().reset()
        sample_idx = self.np_random.randint(len(self.expert_state))
        sample_state = self.expert_state[sample_idx][0]
        print('chosen_goal',self.expert_goals[sample_idx])
        sample_goal = int(self.expert_goals[sample_idx]/self.map.shape[0]), self.expert_goals[sample_idx]%self.map.shape[1]
        sample_goal_orientation = self.expert_goal_orientations[sample_idx] 
        self.set_state(sample_state)
        self.goal_location = sample_goal
        self.goal_orientation = sample_goal_orientation
        obs = self.get_observation_dict() if self.is_goal_env else self.get_observation_dict()['observation']
        return obs
    # def reset(self):
    #     obs = super().reset()
    #     # TODO select a random object and try to pick it up
    #     pickable_objects = [
    #         k for k, v in self.world_objects.items() if v.is_movable]
    #     pickable_objects.remove('agent')
    #     pickable_objects = self.np_random.permutation(pickable_objects)
    #     #pick random empty block
    #     empty_blocks = np.argwhere(obs==0)
    #     self.goal_location = empty_blocks[self.np_random.randint(len(empty_blocks))]
    #     self.goal_orientation =  self.np_random.randint(3)
    #     obs = self.get_observation_dict() if self.is_goal_env else self.get_observation_dict()['observation']
    #     return obs

    # def reset2state(self,state):
    #     obs = super().set_state(state)
    #     # TODO select a random object and try to pick it up
    #     pickable_objects = [
    #         k for k, v in self.world_objects.items() if v.is_movable]
    #     pickable_objects.remove('agent')
    #     pickable_objects = self.np_random.permutation(pickable_objects)
    #     self.goal_object = self.world_objects[pickable_objects[0]]
    #     obs = self.get_observation_dict() if self.is_goal_env else self.get_observation_dict()['observation']
    #     return obs

    # @property
    # def goal_location(self):
    #     return self.goal_location

    @property
    def box_in_front(self):
        agent_direction = self.world_objects['agent'].facing_direction
        agent_direction = [k for k,v in self.directions.items() if v==agent_direction][0]
        if agent_direction in self.agent_neighbors.keys():
            box_in_front = self.agent_neighbors[agent_direction]
        else:
            box_in_front = None
        return box_in_front

    @property
    def achieved_goal_array(self):
        array = np.zeros([self.world_size, self.world_size])
        loc = self.world_objects['agent'].location
        array[loc[0],loc[1]] = 1 + self.world_objects['agent'].matrix_orientation
        # array[loc[0],loc[1],1] = self.world_objects['agent'].matrix_orientation
        return array
    # @property
    # def achieved_goal_array(self):
    #     array = np.zeros([self.world_size, self.world_size, 2])
    #     achieved_location = self.box_in_front
    #     # print('achived location', achieved_location)
    #     if achieved_location is not None:
    #         array[achieved_location[0], achieved_location[1], :] = 1
    #     else:
    #         array[self.world_objects['agent'].location[0],self.world_objects['agent'].location[1],:] = 1
    #     return array

    @property
    def desired_goal_array(self):
        array = np.zeros([self.world_size, self.world_size])
        array[self.goal_location[0], self.goal_location[1]] = 1+ self.goal_orientation
        # array[self.goal_location[0], self.goal_location[1], 1] = self.goal_orientation
        return array

    # @property
    # def achieved_goal_array(self):
    #     array = np.zeros([self.world_size, self.world_size, 2])
    #     achieved_location = self.box_in_front
    #     # print('achived location', achieved_location)
    #     if achieved_location is not None:
    #         array[achieved_location[0], achieved_location[1], :] = 1
    #     else:
    #         array[self.world_objects['agent'].location[0],self.world_objects['agent'].location[1],:] = 1
    #     return array

    # @property
    # def desired_goal_array(self):
    #     array = np.zeros([self.world_size, self.world_size, 2])
    #     array[self.goal_location[0], self.goal_location[1], :] = 1
    #     return array

    @property
    def agent_neighbors(self):
        x, y = self.world_objects['agent'].location
        up = x, y+1
        left = x - 1, y
        down = x, y - 1
        right = x+1, y 
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

    def step(self, action):
        if action == 3:
            action = 4
        obs, rew, done, info = super().step(action)
        obs = self.get_observation_dict()
        rew = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        if not self.is_goal_env:
            obs = obs['observation'] 
        return obs, rew, done, info

    def get_observation_dict(self):
        obs = super().get_observation()
        obs[obs>=2] = 2
        obs = obs[:,:,0]
        return {'observation': obs.reshape(-1), 'achieved_goal': self.achieved_goal_array.reshape(-1), 'desired_goal': self.desired_goal_array.reshape(-1)}

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal = achieved_goal.reshape(self.map.shape[0],self.map.shape[1])
        desired_goal = desired_goal.reshape(self.map.shape[0],self.map.shape[1])
        achieved_goal_location = np.argwhere(np.floor(achieved_goal) == 1)[0]
        desired_goal_location = np.argwhere(np.floor(desired_goal) == 1)[0]
        distance = np.linalg.norm(
            desired_goal_location-achieved_goal_location, 1)
        orientation_diff = abs(achieved_goal[achieved_goal_location[0],achieved_goal_location[1]]- desired_goal[desired_goal_location[0],desired_goal_location[1]])
        orientation_diff = 1 if orientation_diff == 0.75 else orientation_diff*4
        # return distance
        if distance == 0:
            if orientation_diff == 0:
                return 10
            else:
                return 5
        else:
            return 1-distance/(self.map.shape[0]*self.map.shape[1])

    # def compute_reward(self, achieved_goal, desired_goal, info=None):
    #     achieved_goal = achieved_goal.reshape(self.map.shape[0],self.map.shape[1],2)[:,:,0]
    #     desired_goal = desired_goal.reshape(self.map.shape[0],self.map.shape[1],2)[:,:,0]
    #     achieved_goal_location = np.argwhere(achieved_goal == 1)
    #     desired_goal_location = np.argwhere(desired_goal == 1)
    #     if len(achieved_goal_location) == 0 or len(desired_goal_location) == 0:
    #         return 0
    #     distance = np.linalg.norm(
    #         desired_goal_location[0]-achieved_goal_location[0], 1)
    #     # return distance
    #     if distance == 0:
    #         return 10
    #     else:
    #         return 1-distance/(self.map.shape[0]*self.map.shape[1])
