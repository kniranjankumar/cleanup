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
        
    def reset(self):
        obs = super().reset()
        #TODO select a random object and try to pick it up
        pickable_objects = [k for k,v in self.world_objects.items() if v.is_movable]
        pickable_objects.remove('agent')
        pickable_objects = np.random.permute(pickable_objects)
        for object_name in pickable_objects:
            
        return obs
        
    def step(self, action):
        if action == 3:
            action = 4
        obs,rew,done,info = super().step(action)
        return obs,rew,done,info
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        ac