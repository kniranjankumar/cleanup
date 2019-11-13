import pybullet as p
import time
import math
import numpy as np
import os
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
    p.connect(p.GUI)
p.setPhysicsEngineParameter(numSolverIterations=10)
p.setTimeStep(1. / 120.)
# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
# useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
p.loadURDF("plane.urdf", useMaximalCoordinates=True,basePosition=[0.5,0.5,0])
# disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


class CleanupWorld():

    def __init__(self, max_time_steps=100, is_goal_env=True):

        self.action_space_str = ['forward', 'left', 'right', 'pick']
        self.directions = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
        self.done = False
        self.world_size = 8
        # channel 0 for marking object positions and channel 1 to store orientation/direction
        self.map = np.zeros(
            [self.world_size, self.world_size, 2], dtype='uint8')
        self.objectid = {'agent': 1, 'chair': 2}
        self.map[0, 0, 0] = 1  # agent
        self.map[0, 0, 1] = 0
        self.map[3, 3, 0] = 2  # chair
        self.map[3, 3, 1] = 0
        self.world_scale = 1
        pwd = os.getcwd()
        print(pwd)
        self.done = True
        self.agent_location = None
        self.agent_direction = None
        self.item_on_hand = None
        self.item_on_hand_dir = 0
        self.TIME_LIMIT = max_time_steps
        self.t = 0
        self.z = {'chair': 1, 'agent': 0.05}
        self.x = {'chair': 0.35, 'agent': 0.1}
        self.y = {'chair': 0.35, 'agent': 0.05}
        self.offset = {'chair':[-0.35,-0.8,0.3], 'agent':[0, 0, 0.05]}
        
        # self.offset = {'chair':[-0.38, -0.38,0.8], 'agent':[0, 0, 0.05]}
        self.orientation = {'chair': [0.7071068, 0, 0, 0.7071068], 'agent': [
            0, 0.7071068, 0.7071068, 0]}
        self.scale = {'chair': 0.08, 'agent': 0.3}
        self.Uid = []
        self.Uid.append(self.add_mesh(path="./data/lego_man/", mesh_name="LEGO_Man.obj", tex_name=None, position=[
                        0*self.world_scale, 0*self.world_scale, self.z['agent']], orientation=self.orientation['agent'], scale=self.scale['agent']))
        self.Uid.append(self.add_mesh(path="./data/chair/", mesh_name="chair.obj", tex_name="diffuse.tga", position=[
                        3*self.world_scale, 3*self.world_scale, self.z['chair']], orientation=self.orientation['chair'], scale=self.scale['chair']))
        self.put_object_in_grid(self.get_location('agent'),self.get_orientation('agent'),'agent')
        self.put_object_in_grid(self.get_location('chair'),self.get_orientation('chair'),'chair')        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

    def get_object_from_id(self, id):
        for k, v in self.objectid.items():
                            if v == self.item_on_hand:
                                item_name = k
                                return item_name

    def get_location(self, object_name):
        objectid = self.objectid[object_name]
        map_loc = np.argwhere(self.map[:, :, 0] == objectid)[0]
        return map_loc
    
    def get_orientation(self, object_name):
        if self.item_on_hand != None:
            item_name = self.get_object_from_id(self.item_on_hand)
            if object_name == item_name:
                direction = self.item_on_hand_dir
                angle = np.pi/2*direction
                new_angles = [np.pi/2, 0, np.pi+angle]
                new_quaternion = p.getQuaternionFromEuler(new_angles)
                return new_quaternion
        print(object_name)
        objectid = self.objectid[object_name]
        map_loc = np.argwhere(self.map[:, :, 0] == objectid)[0]
        direction = self.map[map_loc[0], map_loc[1],1]
        angle = np.pi/2*direction
        new_angles = [np.pi/2, 0, np.pi+angle]
        new_quaternion = p.getQuaternionFromEuler(new_angles)
        return new_quaternion
        
    def add_mesh(self, path="./data/chair/", mesh_name="chair.obj", tex_name="diffuse.tga", position=[0, 0, 1], orientation=[0.7071068, 0, 0, 0.7071068], scale=0.1):
        shift = [0, -0.02, 0]
        meshScale = [scale, scale, scale]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=path+mesh_name,
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=shift,
                                            meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=path+mesh_name,
                                                  collisionFramePosition=shift,
                                                  meshScale=meshScale)
        rangex = 5
        rangey = 5
        bodyUid = p.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=position,
                                    baseOrientation=orientation,
                                    useMaximalCoordinates=True)
        if tex_name != None:
            texUid = p.loadTexture(path+tex_name)
            p.changeVisualShape(bodyUid, -1, textureUniqueId=texUid)
        return bodyUid

    def reset(self):
        self.map = np.zeros([8, 8], dtype='uint8')
        self.map[0, 0, 0] = 1  # agent
        self.map[3, 3, 0] = 2  # chair

    def put_object_in_grid(self, location, orientation, object_name,on_ground=True):
        objectid = self.objectid[object_name]
        offset = self.offset[object_name]
        R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
        print(R.shape, np.array(offset).shape)
        offset = -R@np.array(offset)
        # base_orientation = self.orientation[object_name]
        if on_ground:
            p.resetBasePositionAndOrientation(objectid, [location[0]*self.world_scale+offset[0], location[1] *
                                                    self.world_scale+offset[1], offset[2]], orientation)  # place agent in new location
        else:
            p.resetBasePositionAndOrientation(objectid, [location[0]*self.world_scale+offset[0], location[1] *self.world_scale+offset[1], 1+offset[2]], orientation)  # place agent in new location

    def update_render(self, object_name, mode='move_agent'):
        sign = [[1, 0], [0, -1], [-1, 0], [0, 1]]

        if mode == 'move_agent':
            objectid = self.objectid[object_name]
            map_loc = self.get_location(object_name)
            Uid = objectid
            new_quaternion = self.get_orientation('agent')
            self.put_object_in_grid(map_loc,new_quaternion,'agent')
            if self.item_on_hand != None:
                # _, orientation = p.getBasePositionAndOrientation(objectid)
                orientation = self.get_orientation(self.get_object_from_id(self.item_on_hand))
                self.put_object_in_grid(map_loc,orientation,self.get_object_from_id(self.item_on_hand),on_ground=False)

        elif mode == 'pick':
            print('picking')
            objectid = self.objectid[object_name]
            map_loc = self.get_location(object_name)
            object_orientation = self.get_orientation(object_name)
            agentid = self.objectid['agent']
            agent_loc = self.get_location('agent')
            Uid = objectid
            agent_orientation = self.get_orientation('agent')
            self.put_object_in_grid(agent_loc,object_orientation,object_name,on_ground=False)
            
        elif mode == 'place':
            objectid = self.objectid[object_name]
            map_loc = self.get_location(object_name)
            Uid = objectid
            new_quaternion = self.get_orientation(object_name)
            self.put_object_in_grid(map_loc,new_quaternion,object_name)
            # if self.item_on_hand != None:
            #     # _, orientation = p.getBasePositionAndOrientation(objectid)
            #     orientation = self.get_orientation(object_name)
            #     self.put_object_in_grid(map_loc,orientation,self.get_object_from_id(self.item_on_hand),on_ground=False)


    def step(self, action):
        map_loc = np.argwhere(self.map[:, :, 0] == self.objectid['agent'])[0]
        if action == 'forward' or action == 'pick':
            print('going', self.map[map_loc[0], map_loc[1], 1])

            if self.map[map_loc[0], map_loc[1], 1] == 0:
                new_loc = map_loc[0], map_loc[1]+1
            elif self.map[map_loc[0], map_loc[1], 1] == 1:
                new_loc = map_loc[0]-1, map_loc[1]
            elif self.map[map_loc[0], map_loc[1], 1] == 2:
                new_loc = map_loc[0], map_loc[1]-1
            elif self.map[map_loc[0], map_loc[1], 1] == 3:
                new_loc = map_loc[0]+1, map_loc[1]

            if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[0] > self.world_size-1 or new_loc[1] > self.world_size-1:
                return
            if action == 'forward':
                if self.map[new_loc[0], new_loc[1], 0] != 0:
                    return

                # print()
                self.map[new_loc[0], new_loc[1], 0] = self.objectid['agent']
                self.map[new_loc[0], new_loc[1],
                         1] = self.map[map_loc[0], map_loc[1], 1]
                self.map[map_loc[0], map_loc[1], :] = 0
                self.update_render('agent')
            elif action == 'pick':
                if self.map[new_loc[0], new_loc[1], 0] == 0:
                    if self.item_on_hand != None: #Execute place action
                        self.map[new_loc[0], new_loc[1], 0] = self.item_on_hand
                        item_name = self.get_object_from_id(self.item_on_hand)
                        self.item_on_hand = None
                        self.map[new_loc[0], new_loc[1], 1] = self.item_on_hand_dir
                        self.item_on_hand_dir = 0
                        print(self.map[new_loc[0], new_loc[1], 0])
                        self.update_render(item_name,mode='place')
                    else:
                        return
                else:
                    if self.item_on_hand != None:
                        return
                    else:
                        self.item_on_hand = self.map[new_loc[0], new_loc[1], 0]
                        self.item_on_hand_dir = self.map[new_loc[0], new_loc[1], 1]
                        item_name = self.get_object_from_id(self.item_on_hand)
                        self.update_render(item_name, mode='pick')
                        self.map[new_loc[0], new_loc[1], 0] = 0

        if action == 'right':
            self.map[map_loc[0], map_loc[1], 1] -= 1
            if self.map[map_loc[0], map_loc[1], 1] == 255:
                self.map[map_loc[0], map_loc[1], 1] = 3
            if self.item_on_hand != None:
                self.item_on_hand_dir -= 1
                if self.item_on_hand_dir < 0:
                    self.item_on_hand_dir = 3 
            self.update_render('agent')
            
        if action == 'left':
            self.map[map_loc[0], map_loc[1], 1] += 1
            if self.map[map_loc[0], map_loc[1], 1] == 4:
                self.map[map_loc[0], map_loc[1], 1] = 0
            if self.item_on_hand != None:
                self.item_on_hand_dir += 1
                if self.item_on_hand_dir == 4:
                    self.item_on_hand_dir = 0 
            self.update_render('agent')
            


if __name__ == "__main__":
    world = CleanupWorld()
    keys = {str(p.B3G_DOWN_ARROW): 'down', str(p.B3G_UP_ARROW): 'up', str(
        p.B3G_RIGHT_ARROW): 'right', str(p.B3G_LEFT_ARROW): 'left', str(32): 'space'}
    while(True):
        keyEvents = p.getKeyboardEvents()
        for k in keyEvents.keys():
            # print(keyEvents[k],p.KEY_WAS_TRIGGERED)
            if keyEvents[k] == 3:
                if str(k) in keys.keys():
                    print(keys[str(k)]+' pressed')
                    if keys[str(k)] == 'up':
                        world.step('forward')
                        # _, orientation = p.getBasePositionAndOrientation(robot)
                        # p.resetBasePositionAndOrientation(robot,pos,orientation)
                    if keys[str(k)] == 'right':
                        world.step('right')
                    if keys[str(k)] == 'left':
                        world.step('left')
                    if keys[str(k)] == 'space':
                        world.step('pick')
                    # pos_current, orientation = p.getBasePositionAndOrientation(robot)
                    # angles = list(p.getEulerFromQuaternion(orientation))
                    # print(angles)
                    # angles[2] -= 1.57
                    # orientation = p.getQuaternionFromEuler(angles)
                    # p.resetBasePositionAndOrientation(robot,pos_current,orientation)


# def getRayFromTo(mouseX, mouseY):

#   width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera(
#   )
#   camPos = [
#       camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
#       camTarget[2] - dist * camForward[2]
#   ]
#   farPlane = 10000
#   rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
#   invLen = farPlane * 1. / (math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] *
#                                       rayForward[1] + rayForward[2] * rayForward[2]))
#   rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
#   rayFrom = camPos
#   oneOverWidth = float(1) / float(width)
#   oneOverHeight = float(1) / float(height)
#   dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
#   dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
#   rayToCenter = [
#       rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
#   ]
#   rayTo = [
#       rayToCenter[0] - 0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] -
#       float(mouseY) * dVer[0], rayToCenter[1] - 0.5 * horizon[1] + 0.5 * vertical[1] +
#       float(mouseX) * dHor[1] - float(mouseY) * dVer[1], rayToCenter[2] - 0.5 * horizon[2] +
#       0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
#   ]
#   return rayFrom, rayTo


# cid = p.connect(p.SHARED_MEMORY)
# if (cid < 0):
#   p.connect(p.GUI)
# p.setPhysicsEngineParameter(numSolverIterations=10)
# p.setTimeStep(1. / 120.)
# logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
# #useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
# p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
# #disable rendering during creation.
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# #disable tinyrenderer, software (CPU) renderer, we don't use it here
# p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


# # for i in range(rangex):
# #   for j in range(rangey):
# #     p.createMultiBody(baseMass=1,
# #                       baseInertialFramePosition=[0, 0, 0],
# #                       baseCollisionShapeIndex=collisionShapeId,
# #                       baseVisualShapeIndex=visualShapeId,
# #                       basePosition=[((-rangex / 2) + i) * meshScale[0] * 2,
# #                                     (-rangey / 2 + j) * meshScale[1] * 2, 1],
# #                       useMaximalCoordinates=True)
# chair = add_mesh(path="./data/chair/",mesh_name="chair.obj",tex_name="diffuse.tga",position=[0,0,1], orientation=[0.7071068,0,0,0.7071068],scale=0.1)
# robot = add_mesh(path="./data/lego_man/",mesh_name="LEGO_Man.obj",tex_name=None,position=[2,0,0.05], orientation=[0.7071068,0,0,0.7071068], scale=0.5)
# pos = [2,0,0.05]
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.stopStateLogging(logId)
# p.setGravity(0, 0, -10)
# p.setRealTimeSimulation(1)

# colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
# currentColor = 0
# keys={str(p.B3G_DOWN_ARROW):'down', str(p.B3G_UP_ARROW):'up',str(p.B3G_RIGHT_ARROW):'right',str(p.B3G_LEFT_ARROW):'left'}
# quit = False
# while (not quit):
#   mouseEvents = p.getMouseEvents()
#   keyEvents = p.getKeyboardEvents()
#   for k in keyEvents.keys():
#     # print(keyEvents[k],p.KEY_WAS_TRIGGERED)
#     if keyEvents[k] == 3:
#       print(keys[str(k)]+' pressed')
#       if keys[str(k)] == 'up':
#         pos[1] -= 1
#         _, orientation = p.getBasePositionAndOrientation(robot)
#         p.resetBasePositionAndOrientation(robot,pos,orientation)
#       if keys[str(k)] == 'right':
#         pos_current, orientation = p.getBasePositionAndOrientation(robot)
#         angles = list(p.getEulerFromQuaternion(orientation))
#         print(angles)
#         angles[2] -= 1.57
#         orientation = p.getQuaternionFromEuler(angles)
#         p.resetBasePositionAndOrientation(robot,pos_current,orientation)
#       # quit=True

#   for e in mouseEvents:
#     if ((e[0] == 2) and (e[3] == 0) and (e[4] & p.KEY_WAS_TRIGGERED)):
#       mouseX = e[1]
#       mouseY = e[2]
#       rayFrom, rayTo = getRayFromTo(mouseX, mouseY)
#       rayInfo = p.rayTest(rayFrom, rayTo)
#       #p.addUserDebugLine(rayFrom,rayTo,[1,0,0],3)
#       for l in range(len(rayInfo)):
#         hit = rayInfo[l]
#         objectUid = hit[0]
#         if (objectUid >= 1):
#           p.changeVisualShape(objectUid, -1, rgbaColor=colors[currentColor])
#           currentColor += 1
#           if (currentColor >= len(colors)):
#             currentColor = 0
