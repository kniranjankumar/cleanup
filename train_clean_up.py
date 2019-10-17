import gym
import cleanup_world
from baselines import deepq
import tensorflow as tf
# from baselines.baselines.common.models import mlp
env = gym.make('2DCleanup-v0')
# env = gym.make('CartPole-v0')
with tf.device('/GPU:0'):
        act = deepq.learn(
                env,
                network='mlp',
                lr=1e-3,
                total_timesteps=100000,
                buffer_size=10000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                print_freq=1)