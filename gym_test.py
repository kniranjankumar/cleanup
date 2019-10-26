import gym
from cleanup_world.envs.clean_up_world_final import CleanupWorld
import random
import pygame as pg
import time

## initialize pygame object
pg.init()

## create the game window
game_window = pg.display.set_mode((480, 480), 0, 0)
pg.display.set_caption('CleanupWorld')

## create the game object
game = CleanupWorld(game_window)

## create an intial drawing and update the game window
game.draw()
pg.display.update()

time.sleep(5)

for i in range(10):
	action = random.choice(game.action_space_str)
	rew, done = game.step(action)
	game.difference()
	print(rew)
	game.draw()
	pg.display.update()
	time.sleep(3)


# pygame.init()
# game_window = pygame.display.set_mode((480, 480))
# pygame.display.set_caption('CleanupWorld')

# env = CleanupWorld(game_window)
# # env.render()
# # obs = env.reset()

# pygame.display.update()
# time.sleep(10)

# print('State Space: ', env.observation_space)
# print('Action Space: ', env.action_space)
# # print('State Space Low: ', env.observation_space.low)
# # print('State Space High: ', env.observation_space.high)

# # image_list = env.return_image_list()
# # print(image_list)

# # while True:
# # 	for event in pygame.event.get():
# # 		if event.type == pygame.K_UP:
# # 			action = 'forward'
# # 		elif event.type == pygame.K_LEFT:
# # 			action = 'left'
# # 		elif event.type == pygame.K_RIGHT:
# # 			action = 'right'
# # 		elif event.type == pygame.K_p:
# # 			action = 'pick'
# # 		elif event.type == pygame.QUIT:
# # 			pygame.quit()
# # 			sys.exit()
# # 		obs, rew, done = env.step(action)
# # 		env.difference()
# # 		print(rew)
# # 		env.render(mode='human')
# # 		pygame.display.update()
# # 		time.sleep(10)

# # for i in range(10):
# # 	## this is where we need to add the human collection of action instead of action
# # 	action = random.choice(env.action_space_str)
# # 	obs, rew, done = env.step(action)
# # 	env.difference()
# # 	print(rew)
# # 	env.render(mode='human')
# # 	pygame.display.update()
# # 	time.sleep(10)
