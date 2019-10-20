import gym
import cleanup_world
# from baselines import deepq
# import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
# from baselines.baselines.common.models import mlp
# env = gym.make('2DCleanup-v0')
# env = gym.make('CartPole-v0')
# with tf.device('/GPU:0'):
#         act = deepq.learn(
#                 env,
#                 network='mlp',
#                 lr=1e-3,
#                 total_timesteps=100000,
#                 buffer_size=10000,
#                 exploration_fraction=0.1,
#                 exploration_final_eps=0.02,
#                 print_freq=1,
#                 gamma=0.98,
#                 prioritized_replay=True)