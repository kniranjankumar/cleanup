import pybullet as p
import time
import math
import numpy as np
import os
import json
import copy
import pickle as pkl
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
p.resetDebugVisualizerCamera( cameraDistance=10, cameraYaw=0,
cameraPitch=-50, cameraTargetPosition=[4,4,0])

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
        # print('relative location', relative_loc)
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
        # print('facing', self.facing_direction)
        angle = np.pi/2*self.facing_direction
        base_angle = p.getEulerFromQuaternion(self.offset_orientation)
        new_angles = [base_angle[0], 0, np.pi+angle]
        quaternion = p.getQuaternionFromEuler(new_angles)
        if self.has_children():
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
        if self.has_children():
            # print(self.children)
            for child_name, child in self.children.items():
                child.put_object_in_grid()

    def has_children(self, loc=None):
        if loc is not None:
            relative_loc = loc[0]-self.location[0], loc[1]-self.location[1]
            if isinstance(self.slots[relative_loc[0],relative_loc[1]],Object):
                return True
        else:
            return False if len(self.children.keys()) == 0 else True
            
    def get_children(self,loc):
        relative_loc = loc[0]-self.location[0], loc[1]-self.location[1]
        if isinstance(self.slots[relative_loc[0],relative_loc[1]],Object):
            return self.slots[relative_loc[0],relative_loc[1]]
        else:
            return None
            
    def local_position(self,loc):
        return loc[0]-self.location[0], loc[1]-self.location[1]
        
    @property
    def has_parent(self):
        return not self.parent == None
    

class CleanupWorld():

    def __init__(self,max_time_steps=100, is_goal_env=True):

        self.action_space_str = ['forward', 'left', 'right', 'pick']
        self.directions = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
        self.done = False
        self.world_size = 8
        # channel 0 for marking object positions and channel 1 to store orientation/direction
        self.map = np.empty((self.world_size,self.world_size), dtype=object)
        self.world_objects = {}
        self.world_properties = {'scale': 1, 'size': self.world_size}
        objects_available = ['agent','chair','plate','tv','couch','book','phone','laptop', 'cupboard','coffee_table']
        # objects_in_world = {'agent':1,'chair':4,'couch':1,'tv':1,'cupboard':1,'coffee_table':1,'laptop':1}
        objects_in_world = {'agent':1}
        # objects_in_world = {}
        with open('objects.json','r') as f:
            obj_list = json.load(f)
        obj_dict = {obj['name']:obj for obj in obj_list}
        # num_objects = 10
        # for i in range(4):
        #     for obj_name, obj in obj_dict.items():
        #         prob = obj['probability']
        #         if np.random.choice([True,False],p=[prob[i], 1-prob[i]]):
        #             if i == 0 or obj_name in objects_in_world.keys():
        #                 if obj_name not in objects_in_world.keys():
        #                     objects_in_world[obj_name] = 1
        #                 else:
        #                     objects_in_world[obj_name] += 1    
        keys = {str(p.B3G_RETURN):'return'}
        for i in range(len(objects_available)-1):
            keys[str(i+49)]= objects[i+1]
        object_count = 0
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        obj_instance = copy.deepcopy(obj_dict['agent'])
        obj_instance['name'] = obj_instance['name']+"_"+str(i)
        loc = self.get_suitable_location(obj_instance['shape'])
        obj = Object(world_properties=self.world_properties, 
            objectid=object_count+1, 
            location=[loc[0], loc[1]],                 
            properties=obj_instance)
        self.map[loc[0]:loc[0]+obj_instance['shape'][0],loc[1]:loc[1]+obj_instance['shape'][1]] = obj
        self.world_objects['agent'] = obj
        object_count += 1
        exit = False
        # text = p.addUserDebugText("Enter the object ids to add the objects:1-chair 2-plate",
        #         [-5,0,5],[0,0,0],1.3)
        for i in range(len(obj_list)-1):
            print(i+1,obj_list[i+1]['name'])
        while True and not exit:        
        # for obj_name,num in objects_in_world.items():            
            # for i in range(num):
            keyEvents = p.getKeyboardEvents()
        # print(keyEvents)
            if len(keyEvents.keys()) != 0:
                for k in keyEvents.keys():
                    if keyEvents[k] == 3:
                        if str(k) in keys.keys():
                            if keys[str(k)] == 'return':
                                exit = True
                                break
                            # print(keys[str(k)]+' selected')
                            print("====================="+keys[str(k)]+' selected'+"======================")
                            # if keys[str(k)] == 'up':
                            obj_name = keys[str(k)]
                            obj_instance = copy.deepcopy(obj_dict[obj_name])
                            if obj_instance['name'] != 'agent':
                                obj_instance['name'] = obj_instance['name']+"_"+str(i)
                            loc = self.get_suitable_location(obj_instance['shape'])
                            if loc is not None:
                                obj = Object(world_properties=self.world_properties, 
                                    objectid=object_count+1, 
                                    location=[loc[0], loc[1]],                 
                                    properties=obj_instance)
                                self.map[loc[0]:loc[0]+obj_instance['shape'][0],loc[1]:loc[1]+obj_instance['shape'][1]] = obj
                                self.world_objects[obj.name] = obj
                                object_count += 1
                                for i in range(len(obj_list)-1):
                                    print(i+1,obj_list[i+1]['name'])
                    
        # for i in range(4):
        #     agent = Object(world_properties=self.world_properties, 
        #         objectid=i+1, 
        #         location=[i, i],                 
        #         properties=obj_list[i])
        #     self.map[i:i+obj_list[i]['shape'][0],i:i+obj_list[i]['shape'][1]] = agent
        #     self.world_objects[obj_list[i]['name']] = agent
            
        self.done = True
        self.item_on_hand = None
        self.TIME_LIMIT = max_time_steps
        self.t = 0
        self.build_the_wall()        
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

    def get_suitable_location(self, shape):
        #get int map from obj map
        int_map = np.zeros_like(self.map,dtype='uint8')
        kernel = np.ones(shape,dtype='uint8')
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                int_map[i,j] = 1 if isinstance(self.map[i,j],Object) else 0
        #convolve int_map with kernel
        conv_out = np.zeros([self.map.shape[0]-kernel.shape[0],self.map.shape[1]-kernel.shape[1]],dtype='int')
        for i in range(self.map.shape[0]-kernel.shape[0]):
            for j in range(self.map.shape[1]-kernel.shape[1]):
                product = int_map[i:i+kernel.shape[0],j:j+kernel.shape[1]]*kernel
                conv_out[i,j] = np.sum(product)
        feasable_locations = np.argwhere(conv_out == 0)
        if len(feasable_locations) == 0:
            print("cannot add more objects")
            return None
        return feasable_locations[np.random.randint(0,feasable_locations.shape[0]),:]
        
    def build_the_wall(self):
        colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[0.01, self.world_size/2, 3])
        visualShapeId = p.createVisualShape(p.GEOM_BOX,halfExtents=[0.1, self.world_size/2, 3])
        bodyUid = p.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=colBoxId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[self.world_size-0.5,self.world_size*0.5-0.5,0],
                                    baseOrientation=[0,0,0,1],
                                    useMaximalCoordinates=True)
        bodyUid = p.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=colBoxId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[-0.5,self.world_size*0.5-0.5,0],
                                    baseOrientation=[0,0,0,1],
                                    useMaximalCoordinates=True)
        bodyUid = p.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=colBoxId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[self.world_size*0.5-0.5,self.world_size-0.5,0],
                                    baseOrientation=[0,0,0.7071068,0.7071068],
                                    useMaximalCoordinates=True)

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

    def get_observation(self):
        int_map = np.zeros([self.map.shape[0],self.map.shape[1],2],dtype='uint8')
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                int_map[i,j,0] = self.map[i,j].objectid if isinstance(self.map[i,j],Object) else 0
                if int_map[i,j,0] != 0:
                    int_map[i,j,1] = self.map[i,j].get_children([i,j]).objectid if self.map[i,j].has_children([i,j]) else 0
        return int_map
        
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
                        # print('adding children')
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
                                    return
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
    objects_in_world = {}
    objects = ['agent','chair','plate','tv','couch','book','phone','laptop', 'cupboard','coffee_table']
    keys = {str(p.B3G_DOWN_ARROW): 'down', str(p.B3G_UP_ARROW): 'up', str(
        p.B3G_RIGHT_ARROW): 'right', str(p.B3G_LEFT_ARROW): 'left', str(32): 'space', str(p.B3G_RETURN):'return'}
    
    # exit = False
    # print("Enter the object ids to add the objects:")
    # while True:
    #     x = input("Item:")
    #     print(x)
    #     if x == 'q':
    #         break
    #     objects_in_world[keys[str(x)]] = 1
    # while True and not exit:
    #     keyEvents = p.getKeyboardEvents()
    #     exit = False
    #     for k in keyEvents.keys():
    #         if keyEvents[k] == 3:
    #             print(keyEvents)
    #             if str(k) in keys.keys():
    #                 print(keys[str(k)]+' added')
    #                 if keys[str(k)] not in objects_in_world:
    #                     objects_in_world[keys[str(k)]] = 1
    #                 else:
    #                     objects_in_world[keys[str(k)]] += 1
    #                 if keys[str(k)] == 'space':
    #                     exit = True
    #                     break
                    
    world = CleanupWorld()
    data = []    
    exit = False
    DIR = './collected_data/'
    try:
        num_files=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])    
    except:
        os.mkdir('./collected_data')
        num_files=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])    
        
    while(True) and not exit:
        keyEvents = p.getKeyboardEvents()
        # print(keyEvents)
        for k in keyEvents.keys():
            if keyEvents[k] == 3:
                if str(k) in keys.keys():
                    # print(keys[str(k)]+' pressed')
                    if keys[str(k)] == 'up':
                        action = 'forward'
                    if keys[str(k)] == 'right':
                        action = 'right'
                    if keys[str(k)] == 'left':
                        action = 'left'
                    if keys[str(k)] == 'space':
                        action = 'pick'
                    if keys[str(k)] == 'return':
                        exit = True
                        break
                    world.step(action)
                    data.append({'obs':world.get_observation(),'action':action})
    print('num_files',num_files)
    with open('collected_data/eps_'+str(num_files)+'.pkl', 'wb') as f:
        pkl.dump(data, f)
    
    
