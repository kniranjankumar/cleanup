import cv2
import numpy as np
img = cv2.imread('./images/grass.png')
cv2.imwrite('bg.png', cv2.resize(img,(32,32)))
# print(img.shape)
# for i in range(32):
#     for j in range(32):
#         cv2.imwrite('./images/objects/img_'+str(i)+'_'+str(j)+'.png',img[i*32:(i+1)*32,j*32:(j+1)*32,:])