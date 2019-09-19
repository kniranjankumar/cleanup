import numpy as np
import cv2

moves = ['Left', 'Right', 'Up', 'Down']
layout_map = {}

def split_img(image, layout_matrix):
    horiz = np.split(image,indices_or_sections=layout_matrix.shape[0], axis=0)
    vert = [np.split(image_strip,indices_or_sections=layout_matrix.shape[0], axis=1) for image_strip in horiz]
    for i in range(layout_matrix.shape[0]):
        for j in range(layout_matrix.shape[1]):
            layout_map[str(i*4+j)] = vert[i][j].shape

def generate_random_layout(grid):
    choose_move = moves[np.random.randint(0,3)]
    # print(choose_move)
    max_idx = grid.argmax()
    max = np.max(grid)
    grid = grid.reshape(-1)
    # print(int(max_idx/4)+1)
    if choose_move == 'Right':
        if int(max_idx % 4) + 1 == grid_shape[1]:
            print('no right')
            pass
        else:
            temp = grid[max_idx+1]
            grid[max_idx+1] = max
            grid[max_idx] = temp
    if choose_move == 'Down':
        if max_idx > grid.shape[0] * (grid_shape[1] - 1):
            print('no down')
            pass
        else:
            temp = grid[max_idx+grid_shape[1]]
            grid[max_idx+grid_shape] = max
            grid[max_idx] = temp
    if choose_move == 'Left':
        if int(max_idx % 4) == 0:
            print('no left')
            pass
        else:
            temp = grid[max_idx-1]
            grid[max_idx-1] = max
            grid[max_idx] = temp
    if choose_move == 'Up':
        if max_idx < grid_shape[0]:
            print('no up')
            pass
        else:
            temp = grid[max_idx-grid_shape[1]]
            grid[max_idx-grid_shape] = max
            grid[max_idx] = temp

    return choose_move, grid.reshape(grid_shape)

def shuffle_image(image, grid):
    img_list = split_img(image,grid)
    # print(len(img_list))

grid_shape = [4, 4]
size = np.prod(grid_shape)
grid = np.linspace(1,size,size)
grid = grid.reshape(grid_shape).astype('uint8')
image = cv2.imread('bee.jpg')
shuffle_image(image,grid)
# grid[0,0] = 20
# for i in range(10):
#     action, grid = generate_random_layout(grid)
#     print(action)
#     print(grid)