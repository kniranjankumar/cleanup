import pybullet as p
import time
import math
import numpy as np
import os
import json
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
    def __init__(self, world_properties, objectid, properties,facing_direction=0, location=[0,0]):
        self.offset_position=properties['offset_position'] 
        self.offset_orientation=properties['offset_orientation']
        self.scale=properties['scale']
        self.surface_offset=properties['surface_offset']
        self.is_infertile=properties['is_infertile']
        self.name = properties['name']       
        self.world_properties = world_properties
        self.objectid = objectid
        self.facing_direction = facing_direction
        self.orientation = properties['offset_orientation']
        self.location = location
        self.children = {}
        self.parent = None
        shape = properties['shape']
        self.slots = np.empty(shape, dtype=object)
        self.is_movable = properties['is_movable']
        self.Uid = self.add_mesh(path=properties['path'], mesh_name=properties['model_name'], tex_name=properties['texture_name'], position=[
            location[0]*self.world_properties['scale'], location[1]*self.world_properties['scale'], 1], orientation=self.offset_orientation, scale=self.scale)
        self.put_object_in_grid()

    def add_children(self, child,loc):
        assert self.is_infertile == False
        relative_loc = self.local_position(loc)
        print('relative location', relative_loc)
        if isinstance(self.slots[relative_loc[0],relative_loc[1]],Object):
            return False
        else:
            self.slots[relative_loc[0],relative_loc[1]] = child
            self.children[child.name] = child
            child.set_location(loc)
            child.parent = self
            return True
        # self.is_infertile = True

        
    def remove_children(self,loc):
        relative_loc = self.local_position(loc)
        if isinstance(self.slots[relative_loc[0],relative_loc[1]],Object):
            child = self.slots[relative_loc[0],relative_loc[1]]
            del self.children[child.name]
            child.parent = None
            self.slots[relative_loc[0],relative_loc[1]] = object()
        return child
        
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
        base_angle = p.getEulerFromQuaternion(self.offset_orientation)
        new_angles = [base_angle[0], 0, np.pi+angle]
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
        path1 = "./data/chair/"
        mesh_name1="chair.obj"
        tex_name1=None
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

    def put_object_in_grid(self):
        R = np.array(p.getMatrixFromQuaternion(self.orientation)).reshape(3, 3)
        # print(R.shape, np.array(offset).shape)
        offset = -R@np.array(self.offset_position)
        # base_orientation = self.orientation[object_name]
        if self.has_parent:
            height = self.parent.surface_offset
        else:
            height = 0
        p.resetBasePositionAndOrientation(self.objectid, [self.location[0]*self.world_properties['scale']+offset[0], self.location[1]
                                                              * self.world_properties['scale']+offset[1], height+offset[2]], self.orientation)  # place agent in new location
        if self.has_children:
            # print(self.children)
            for child_name, child in self.children.items():
                child.put_object_in_grid()

    def has_children(self, loc=[0,0]):
        relative_loc = loc[0]-self.location[0], loc[1]-self.location[1]
        
        return False if len(self.children.keys()) == 0 else True
  
    def local_position(self,loc):
        return loc[0]-self.location[0], loc[1]-self.location[1]
        
    @property
    def has_parent(self):
        return not self.parent == None
    

class CleanupWorld():

    def __init__(self, max_time_steps=100, is_goal_env=True):

        self.action_space_str = ['forward', 'left', 'right', 'pick']
        self.directions = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
        self.done = False
        self.world_size = 8
        # channel 0 for marking object positions and channel 1 to store orientation/direction
        self.map = np.empty((self.world_size,self.world_size), dtype=object)
        self.world_objects = {}
        self.world_properties = {'scale': 1, 'size': 8}
        with open('objects.json','r') as f:
            obj_list = json.load(f)
        for i in range(len(obj_list)):
            agent = Object(world_properties=self.world_properties, 
                objectid=i+1, 
                location=[i, i],                 
                properties=obj_list[i])
            self.map[i:i+obj_list[i]['shape'][0],i:i+obj_list[i]['shape'][1]] = agent
            self.world_objects[obj_list[i]['name']] = agent
            
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
        
    def get_obj_from_id(self, id):
        for k, v in self.objectid.items():
            if v == id:
                obj_name = k
                return obj_name

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
            # print('here', new_loc)

            if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[0] > self.world_properties['size']-1 or new_loc[1] > self.world_properties['size']-1:
                return
            if action == 'forward':
                if isinstance(self.map[new_loc[0], new_loc[1]],Object):
                    return

                self.map[new_loc[0], new_loc[1]] = self.world_objects['agent']
                self.world_objects['agent'].set_location(new_loc)
                # self.world_objects['agent'].location = new_loc[0], new_loc[1]
                self.map[map_loc[0], map_loc[1]] = object()
                self.update_render(self.world_objects['agent'])
            elif action == 'pick':
                if not self.world_objects['agent'].has_children():               
                    if isinstance(self.map[new_loc[0], new_loc[1]],Object):
                        # Execute pick action
                        obj_in_front = self.map[new_loc[0], new_loc[1]]
                        if obj_in_front.has_children(new_loc):
                            obj_in_front = obj_in_front.remove_children(new_loc)
                        else:
                            if obj_in_front.is_movable:
                                self.map[new_loc[0], new_loc[1]] = object()  
                            else:
                                return                          
                        print('adding children')
                        self.world_objects['agent'].add_children(obj_in_front,self.world_objects['agent'].location)
                        self.update_render(self.world_objects['agent'])
                        # self.map[new_loc[0], new_loc[1]] = 0
                else:
                    # Execute place action
                        obj_in_front = self.map[new_loc[0], new_loc[1]]
                        if isinstance(obj_in_front,Object): 
                            if not obj_in_front.is_infertile:
                                child = self.world_objects['agent'].remove_children(self.world_objects['agent'].location)                                
                                success = obj_in_front.add_children(child,new_loc)
                                if not success:
                                    self.world_objects['agent'].add_children(child,self.world_objects['agent'].location)
                            else:
                                return
                        else:
                            child = self.world_objects['agent'].remove_children(self.world_objects['agent'].location)                        
                            self.map[new_loc[0], new_loc[1]] = child
                        # del self.world_objects['agent'].children[child.name]
                        child.set_location(new_loc)
                        self.update_render(child)


        if action == 'right':
            self.world_objects['agent'].set_orientation('right')
            self.update_render(self.world_objects['agent'])

        if action == 'left':
            self.world_objects['agent'].set_orientation('left')
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
