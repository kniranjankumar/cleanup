from cleanup_world.envs.clean_up_world import CleanupWorld
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from gym.spaces import Dict, Box, Discrete


class PickupWorld(CleanupWorld):
    def __init__(self, max_time_steps=100, is_goal_env=True, render=False):
        super().__init__(max_time_steps=max_time_steps, is_goal_env=is_goal_env, render=render)
        self.action_space = Discrete(4)
        vector_box = Box(
            high=np.ones([self.world_size * self.world_size * 2]),
            low=-1 * np.ones([self.world_size * self.world_size * 2]),
            dtype="float",
        )
        if self.is_goal_env:
            self.observation_space = Dict(
                {
                    "observation": vector_box,
                    "achieved_goal": vector_box,
                    "desired_goal": vector_box,
                }
            )
        
    def reset(self):
        obs = super().reset()
        #TODO select a random object and try to pick it up
        pickable_objects = [k for k,v in self.world_objects.items() if v.is_movable]
        pickable_objects.remove('agent')
        pickable_objects = self.np_random.permute(pickable_objects)
        self.goal_object = self.world_objects(pickable_objects[0])
        return obs
        
    @property    
    def goal_location(self):
        return self.goal_object.location
        
    @property
    def box_in_front(self):
        if self.world_objects['agent'].direction in self.neighbors.keys():
            box_in_front = self.agent_neighbors[self.world_objects['agent'].direction]
        else:
            box_in_front = None
        return box_in_front
        
    @property
    def achieved_goal_array(self):
        array = np.zeros([self.world_size[0],self.world_size[1,2]])
        achieved_location = self.box_in_front
        if achieved_location is not None:
            array[achieved_location,achieved_location,:] = 1
        return array
    @property
    def desired_goal_array(self):
        array = np.zeros([self.world_size[0],self.world_size[1,2]])
        array[self.goal_location[0],self.goal_location[1],:] = 1
        return array
        
    @property
    def agent_neighbors(self):
        x, y =  self.world_objects['agent'].location
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
        

    
    def step(self, action):
        if action == 3:
            action = 4
        obs,rew,done,info = super().step(action)
        return obs,rew,done,info
    
    def get_observation(self):
        obs = super().get_observation()
        return {'observation':obs, 'achieved_goal':self.achieved_goal_array,'desired_goal':self.desired_goal_array}
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal_location = np.argwhere(achieved_goal==1)
        desired_goal_location = np.argwhere(desired_goal==1)
        if len(achieved_goal_location)==0 or len(desired_goal_location)==0:
            return 0