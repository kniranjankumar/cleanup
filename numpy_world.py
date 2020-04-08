import numpy as np
import cv2
from matplotlib import pyplot as plt
from gym import register
import random
import os
from copy import deepcopy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Dict, Box, Discrete

import os
from stable_baselines import HER, DQN, A2C
from options import Parser
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
# from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, nature_cnn
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.deepq.policies import FeedForwardPolicy as FeedForwardPolicy_DQN
from options import Parser


class PickupMlpPolicyDQN(FeedForwardPolicy_DQN):
    """
    Policy object that implements DQN policy, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(PickupMlpPolicyDQN, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling, layers=[512,512,128,64],
                                        layer_norm=False, **_kwargs)
                                        
class NumpyWorld(gym.Env):
    def __init__(self, world_size=[8,8], num_objects=2, max_time_steps=2):
        self.world_size = world_size
        self.num_objects = num_objects
        self.action_space = Discrete(self.world_size[0]*self.world_size[1])
        self.observation_space = Dict(
            {'observation': Box(high=num_objects * np.ones(self.world_size)[-1], low=np.zeros(self.world_size)[-1], dtype='uint8'),
             'achieved_goal': Box(high=num_objects*np.ones(self.world_size)[-1] , low=np.zeros(self.world_size)[-1], dtype='uint8'),
             'desired_goal': Box(high=num_objects*np.ones(self.world_size)[-1] , low=np.zeros(self.world_size)[-1], dtype='uint8')})
        self.world_state = np.zeros(self.world_size)
        self.goal_state = np.zeros(self.world_size)
        self.max_time_steps = max_time_steps
        self.time_step = 0
        self.hand = 0
        
    def reset(self):
        self.time_step = 0
        self.hand = 0
        self.world_state = self.generate_random_layout()
        self.goal_state = self.generate_random_layout()
        
    def generate_random_layout(self):
        state =  np.zeros(self.world_size)
        positions_encoded = np.random.randint(0, self.world_size[0]*self.world_size[1], self.num_objects)
        positions = np.vstack((positions_encoded//self.world_size[0], positions_encoded%self.world_size[1]))
        positions = list(positions)
        for i in range(len(positions)):
            state[positions[i][0], positions[i][1]] = i+1
        # print(self.world_state)
        return state
         
    def step(self, action):
        assert self.time_step < self.max_time_steps
        position = action // self.world_size[0], action % self.world_size[0]
        temp = deepcopy(self.hand)
        self.hand = self.world_state[position[0], position[1]]
        self.world_state[position[0], position[1]] = temp
        self.time_step += 1
        if self.time_step < self.max_time_steps:
            done = False
        else:
            done = True
        obs = self.get_obs()
        rew = self.compute_reward(obs['achieved_goal'],obs['desired_goal'], None)
        return obs, rew, done, {}
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = 0
        for i in range(self.num_objects):
            obj_desired = np.argwhere(desired_goal==i+1)
            obj_achieved = np.argwhere(achieved_goal==i+1)
            if len(obj_desired) != 0 and len(obj_achieved) != 0:
                distance += np.linalg.norm(obj_desired - obj_achieved)
            else:
                distance +=self.world_size[0]
        return -distance
        
    def get_obs(self):
        return {'observation':self.world_state[-1], 'achieved_goal':self.world_state[-1], 'desired_goal':self.goal_state[-1]}
        
    def render(self):
        obs = self.get_obs()
        plt.matshow(np.hstack((obs['achieved_goal'], 10*np.ones([self.world_size[1],1]),obs['desired_goal'])))
        plt.pause(1)
        
if __name__ == "__main__":
    register(
    id='2DNumpyWorld-v1',
    entry_point='numpy_world:NumpyWorld')
    env = gym.make('2DNumpyWorld-v1')
    parser = Parser('DQN_HER')
    args = parser.parse()
    model = HER(PickupMlpPolicyDQN, 
            env, 
            DQN, 
            n_sampled_goal=args.num_sampled_goals, 
            goal_selection_strategy=args.goal_selection_strategy,
            verbose=args.verbose, 
            exploration_fraction=args.exploration_fraction,                 
            tensorboard_log='/srv/share/nkannabiran3/numpy_world/DQN_HER',
            double_q=args.double_q,
            gamma=args.gamma,
            exploration_final_eps=args.exploration_final_eps,
            train_freq=args.train_freq,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            target_network_update_freq=args.target_network_update_freq,
            prioritized_replay=args.prioritized_replay,
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            prioritized_replay_beta0=args.prioritized_replay_beta0,
            prioritized_replay_beta_iters=args.prioritized_replay_beta_iters,
            prioritized_replay_eps=args.prioritized_replay_eps,
            param_noise=args.param_noise,
            learning_rate=args.learning_rate)
    model.learn(total_timesteps=args.num_learning_steps,tb_log_name=args.tensorboard_log_name)
    
    # rew_list = []
    # for i in range(100):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         print(obs)
    #         obs, rew, done,_ = env.step(env.action_space.sample())
    #         env.render()
    #         rew_list.append(rew)
    #     print(obs, rew)
    # print(env.observation_space.sample())