import gym
import cleanup_world
import os
from stable_baselines import HER, DQN
from options import Parser
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
# from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, nature_cnn
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.deepq.policies import FeedForwardPolicy
from options import Parser
parser = Parser("DQN")
args = parser.parse()
def custom_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    # layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    
class PickupCnnPolicy(FeedForwardPolicy):
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
        super(PickupCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling, cnn_extractor=custom_cnn,
                                        layer_norm=False, **_kwargs)


env = gym.make('2DPickup-v0')
# gamma=0.99
# learning_rate=5e-4
# buffer_size=50000
# exploration_fraction=0.1
# exploration_final_eps=0.02
# train_freq=1
# batch_size=32
# double_q=True
# learning_starts=1000
# target_network_update_freq=500
# prioritized_replay=False
# prioritized_replay_alpha=0.6
# prioritized_replay_beta0=0.4
# prioritized_replay_beta_iters=None
# prioritized_replay_eps=1e-6
# param_noise=False
# n_cpu_tf_sess=None
# verbose=0

model = DQN(PickupCnnPolicy, 
            env, 
            verbose=1,
            tensorboard_log='/srv/share/nkannabiran3/DQN/',
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
            param_noise=args.param_noise)

model.learn(total_timesteps=args.num_learning_steps)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    # env.render()