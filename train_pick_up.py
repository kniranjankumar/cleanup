import gym
import cleanup_world
import os
from stable_baselines import HER, DQN
from options import Parser
# from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
# parser = Parser()
# args = parser.parse()
# print(vars(args))
env = gym.make('2DPickup-v0')
model = DQN(MlpPolicy, env, verbose=1,tensorboard_log='/srv/share/nkannabiran3/DQN/',
            double_q=True)#,
            # prioritized_replay=True,
            # prioritized_replay_alpha=0.8,
            # prioritized_replay_beta0=0.2)
model.learn(total_timesteps=100000)
# if args.enable_HER:
#     model = HER('MlpPolicy', 
#                 env, 
#                 DQN, 
#                 n_sampled_goal=args.num_sampled_goals, 
#                 goal_selection_strategy=args.goal_selection_strategy,
#                 verbose=1, 
#                 exploration_fraction=args.exploration_fraction, 
#                 tensorboard_log=args.tensorboard_log_path+'/'+args.name)
# else:
#     model = DQN(MlpPolicy, env, verbose=1,tensorboard_log='/srv/share/nkannabiran3/DQN/',
#             double_q=True,
#             prioritized_replay=True,
#             prioritized_replay_alpha=0.8,
#             prioritized_replay_beta0=0.2)
# print('learning')
# os.mkdir(args.tensorboard_log_path+'/'+args.name)
# parser.save_args()
# model.learn(total_timesteps=args.num_training_steps, tb_log_name=args.tensorboard_log_path+'/'+args.name)
# model.save(args.name)


# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    # env.render()