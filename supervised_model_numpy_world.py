import torch
import numpy as np
import gym
from gym import register
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
register(
id='2DNumpyWorld-v1',
entry_point='numpy_world:NumpyWorld')
env = gym.make('2DNumpyWorld-v1')
class CleanupDataset(Dataset):
    def __init__(self, observations,labels,transform=None,is_train=True, valid_idx=None):
        self.obs = observations
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        obs, action = self.obs[idx].reshape(-1), self.labels[idx]
        if self.transform == None:
            return {'obs':obs, 'label':action}
        else:
            return self.transform({'obs':obs, 'label':action})
  
    def __len__(self,):
        return int(len(self.obs))

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        obs, label = sample['obs'], sample['label']
        obs, label = torch.from_numpy(obs).type(torch.FloatTensor), torch.tensor(label)
        return {'obs':obs.to(device), 'label':label.to(device)}
    
def get_expert_action(achieved_goal, desired_goal):
        distance = {}
        for i in range(env.num_objects):
            obj_desired = np.argwhere(desired_goal==i+1)
            obj_achieved = np.argwhere(achieved_goal==i+1)
            if len(obj_desired) != 0 and len(obj_achieved) != 0:
                distance[str(i+1)]= np.linalg.norm(obj_desired - obj_achieved)
            else:
                distance[str(i+1)] = -1
        if -1 in distance.values():
            object_num = int([k for k, v in distance.items() if v == -1][0])
            obj_desired = np.argwhere(desired_goal==object_num)
            
        
def collect_data(num_eps):
    for eps in num_eps:
        done = False
        obs = env.reset()
        while not done:
            print(obs)
            obs, rew, done,_ = env.step(env.action_space.sample())
        print(obs, rew)
    print(env.observation_space.sample())
