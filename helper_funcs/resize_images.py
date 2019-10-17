import cv2
import os

image_directory = '/home/nithin/Desktop/cleanup/2D_cleanup/cleanup_world/envs/images/scene_objects/'

width = 32
height = 32

for filename in os.listdir(image_directory):
	image_file = os.path.join(image_directory,filename)
	original_image = cv2.imread(image_file)
	new_image = cv2.resize(original_image, (width, height))
	cv2.imwrite(image_file, new_image)
