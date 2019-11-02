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
p.loadURDF("plane.urdf", useMaximalCoordinates=True,
           basePosition=[0.5, 0.5, 0])
# disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


class Object():
    def __init__(self, world_properties, objectid, facing_direction, name, offset_position, offset_orientation, location, scale, mesh):
        self.world_properties = world_properties
        self.objectid = objectid
        self.facing_direction = facing_direction
        self.name = name
        self.offset_position = offset_position
        self.offset_orientation = offset_orientation
        self.orientation = self.offset_orientation

        self.location = location
        self.scale = scale
        self.children = {}
        self.parent = None
        self.Uid = self.add_mesh(path=mesh['path'], mesh_name=mesh['model_name'], tex_name=mesh['texture_name'], position=[
            location[0]*self.world_properties['scale'], location[1]*self.world_properties['scale'], 1], orientation=self.offset_orientation, scale=self.scale)
        self.put_object_in_grid()

    def add_children(self, child):
        self.children[child.name] = child
        child.set_location(self.location)

    def set_orientation(self, turn_direction):
        if turn_direction == 'right':
            self.facing_direction -= 1
            if self.facing_direction == -1:
                self.facing_direction = 3
        if turn_direction == 'left':
            self.facing_direction += 1
            if self.facing_direction == 4:
                self.facing_direction = 0
        print('facing', self.facing_direction)
        angle = np.pi/2*self.facing_direction
        new_angles = [np.pi/2, 0, np.pi+angle]
        quaternion = p.getQuaternionFromEuler(new_angles)
        if self.has_children:
            for child_name, child in self.children.items():
                child.set_orientation(turn_direction)
        self.orientation = quaternion

    def set_location(self, loc):
        self.location = loc
        if self.has_children:
            for child_name, child in self.children.items():
                child.set_location(loc)

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

    def put_object_in_grid(self, on_ground=True):
        R = np.array(p.getMatrixFromQuaternion(self.orientation)).reshape(3, 3)
        # print(R.shape, np.array(offset).shape)
        offset = -R@np.array(self.offset_position)
        # base_orientation = self.orientation[object_name]
        if on_ground:
            p.resetBasePositionAndOrientation(self.objectid, [self.location[0]*self.world_properties['scale']+offset[0], self.location[1] *
                                                              self.world_properties['scale']+offset[1], offset[2]], self.orientation)  # place agent in new location
        else:
            p.resetBasePositionAndOrientation(self.objectid, [self.location[0]*self.world_properties['scale']+offset[0], self.location[1]
                                                              * self.world_properties['scale']+offset[1], 1+offset[2]], self.orientation)  # place agent in new location
        if self.has_children:
            print(self.children)
            for child_name, child in self.children.items():
                child.put_object_in_grid(on_ground=False)

    @property
    def has_children(self):
        print(self.children)

        return False if len(self.children.keys()) == 0 else True

    def has_parent(self):
        return False if self.parent == None else True


class CleanupWorld():

    def __init__(self, max_time_steps=100, is_goal_env=True):

        self.action_space_str = ['forward', 'left', 'right', 'pick']
        self.directions = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
        self.done = False
        self.world_size = 8
        # channel 0 for marking object positions and channel 1 to store orientation/direction
        self.map = np.zeros(
            [self.world_size, self.world_size], dtype='uint8')
        self.objectid = {'agent': 1, 'chair': 2}
        self.object_list = []
        self.world_properties = {'scale': 1, 'size': 8}
        agent_mesh = {'path': "./data/lego_man/",
                      'model_name': "LEGO_Man.obj", 'texture_name': None}
        agent = Object(world_properties=self.world_properties, objectid=1, facing_direction=0, name='agent', offset_position=[
                       0, 0, 0.05], offset_orientation=[0, 0.7071068, 0.7071068, 0], location=[0, 0], scale=0.3, mesh=agent_mesh)
        chair_mesh = {'path': "./data/chair/",
                      'model_name': "chair.obj", 'texture_name': "diffuse.tga"}
        chair = Object(world_properties=self.world_properties, objectid=2, facing_direction=0, name='chair', offset_position=[
                       -0.35, -0.8, 0.3], offset_orientation=[0.7071068, 0, 0, 0.7071068], location=[3, 3], scale=0.08, mesh=chair_mesh)
        self.world_objects = {'agent': agent, 'chair': chair}
        self.map[0, 0] = agent.objectid
        self.map[3, 3] = chair.objectid
        self.done = True
        self.item_on_hand = None
        self.TIME_LIMIT = max_time_steps
        self.t = 0
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

    def reset(self):
        self.map = np.zeros([8, 8], dtype='uint8')
        self.map[0, 0] = self.world_objects['agent'].objectid  # agent
        self.world_objects['agent'].set_location([0,0])
        self.map[3, 3] = self.world_objects['chair'].objectid
        self.world_objects['chair'].set_location([3,3])
        

    def update_render(self, obj):
        obj.put_object_in_grid()

    def step(self, action):
        map_loc = self.world_objects['agent'].location
        if action == 'forward' or action == 'pick':
            if self.world_objects['agent'].facing_direction == 0:
                new_loc = map_loc[0], map_loc[1]+1
            elif self.world_objects['agent'].facing_direction == 1:
                new_loc = map_loc[0]-1, map_loc[1]
            elif self.world_objects['agent'].facing_direction == 2:
                new_loc = map_loc[0], map_loc[1]-1
            elif self.world_objects['agent'].facing_direction == 3:
                new_loc = map_loc[0]+1, map_loc[1]
            print('here', new_loc)

            if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[0] > self.world_properties['size']-1 or new_loc[1] > self.world_properties['size']-1:
                return
            if action == 'forward':
                if self.map[new_loc[0], new_loc[1]] != 0:
                    return

                self.map[new_loc[0], new_loc[1]
                         ] = self.world_objects['agent'].objectid
                self.world_objects['agent'].set_location(new_loc)
                # self.world_objects['agent'].location = new_loc[0], new_loc[1]
                self.map[map_loc[0], map_loc[1]] = 0
                self.update_render(self.world_objects['agent'])
            elif action == 'pick':
                if self.map[new_loc[0], new_loc[1]] == 0:
                    # Execute place action
                    if self.world_objects['agent'].has_children:
                        child = list(
                            self.world_objects['agent'].children.values())[0]
                        self.map[new_loc[0], new_loc[1]] = child.objectid
                        del self.world_objects['agent'].children[child.name]
                        child.set_location(new_loc)
                        print(self.map[new_loc[0], new_loc[1]])
                        self.update_render(child)
                    else:
                        return
                else:
                    if self.world_objects['agent'].has_children != False:
                        return
                    else:
                        obj_in_front_name = self.get_obj_from_id(
                            self.map[new_loc[0], new_loc[1]])
                        self.world_objects['agent'].add_children(
                            self.world_objects[obj_in_front_name])
                        self.update_render(self.world_objects['agent'])
                        self.map[new_loc[0], new_loc[1]] = 0

        if action == 'right':
            self.world_objects['agent'].set_orientation('right')
            if self.item_on_hand != None:
                self.item_on_hand_dir -= 1
                if self.item_on_hand_dir < 0:
                    self.item_on_hand_dir = 3
            self.update_render(self.world_objects['agent'])

        if action == 'left':
            self.world_objects['agent'].set_orientation('left')
            if self.item_on_hand != None:
                self.item_on_hand_dir += 1
                if self.item_on_hand_dir == 4:
                    self.item_on_hand_dir = 0
            self.update_render(self.world_objects['agent'])


if __name__ == "__main__":
    world = CleanupWorld()
    keys = {str(p.B3G_DOWN_ARROW): 'down', str(p.B3G_UP_ARROW): 'up', str(
        p.B3G_RIGHT_ARROW): 'right', str(p.B3G_LEFT_ARROW): 'left', str(32): 'space'}
    while(True):
        keyEvents = p.getKeyboardEvents()
        for k in keyEvents.keys():
            if keyEvents[k] == 3:
                if str(k) in keys.keys():
                    print(keys[str(k)]+' pressed')
                    if keys[str(k)] == 'up':
                        world.step('forward')
                   if keys[str(k)] == 'right':
                        world.step('right')
                    if keys[str(k)] == 'left':
                        world.step('left')
                    if keys[str(k)] == 'space':
                        world.step('pick')
