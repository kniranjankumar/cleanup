import gym
from cleanup_world.envs.clean_up_world import CleanupWorld
import random
import cv2

env = CleanupWorld()
obs = env.reset()

print('State Space: ', env.observation_space)
print('Action Space: ', env.action_space)
# print('State Space Low: ', env.observation_space.low)
# print('State Space High: ', env.observation_space.high)

# image_list = env.return_image_list()
# print(image_list)

for i in range(1):
	action = random.choice(env.action_space_str)
	obs, rew, done = env.step(action)
	env.difference()
	print(rew)
	# cv2.imwrite('env_gif/img_'+str(i).zfill(3)+'.png',obs['observed'])
	env.render(mode='human')
	cv2.imshow('win', obs)
	cv2.waitKey(100)
