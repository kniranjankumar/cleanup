import gym
import cleanup_world
from baselines import deepq
# from baselines.baselines.common.models import mlp
env = gym.make('2DCleanup-v0')
act = deepq.learn(
        env,
        network='cnn',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10)