# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import torch
from torch import nn
import torchvision
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm, tqdm_notebook
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
with open('.clean_data/combined.pkl','rb') as f:
    data = pkl.load(f)
len(data)


# %%
observations = []
labels = []
actions = {'forward':0, 'left':1, 'right':2, 'pick':3}
for item in data:
    observations.append(item['obs'])
    labels.append(actions[item['action']])
# observations = np.array(observations)
# labels = np.array(labels)


# %%
def noramlize(data):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    var = np.std(data, axis=0)
    return (data-mean)/(var+1e-100)

norm_observations = list(noramlize(observations))


# %%
class CleanupDataset(Dataset):
    def __init__(self, observations,labels,transform=None,is_train=True, valid_idx=None):
        self.obs = observations
        self.labels = labels
        self.actions = {'forward':1, 'left':2, 'right':3, 'pick':4}
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


# %%
shuffled_idx = np.random.permutation(len(observations))
train_idx = shuffled_idx[:int(len(shuffled_idx)*0.8)]
test_idx = shuffled_idx[int(len(shuffled_idx)*0.8):]
train_obs = [norm_observations[idx] for idx in train_idx]
train_labels = [labels[idx] for idx in train_idx]
test_obs = [norm_observations[idx] for idx in test_idx]
test_labels = [labels[idx] for idx in train_idx]
print(len(train_obs), len(test_obs))

cleanuptrain = CleanupDataset(observations=train_obs, labels=train_labels,transform=ToTensor())
cleanuptest = CleanupDataset(observations=test_obs, labels=test_labels,transform=ToTensor())
train_loader = DataLoader(cleanuptrain, batch_size=16, shuffle=True)
test_loader = DataLoader(cleanuptest, batch_size=16)


# %%
class SimpleFCN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        #64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(256, 4)
    
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.fc1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


# %%
model = SimpleFCN()
model.train()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
loss_list = []
num_epochs = 100
for i in tqdm(range(num_epochs)):
    for idx, sample in enumerate(train_loader):
        out = model(sample['obs'])
        loss = criterion(out,sample['label'])
        if idx%100 ==0:
            loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#     break


# %%
plt.plot(loss_list)


# %%
from scipy.ndimage.filters import gaussian_filter1d
y = np.array(loss_list)
ysmoothed = gaussian_filter1d(y, sigma=2)
plt.plot(ysmoothed)
plt.xlabel('x 100 training steps')
plt.ylabel('Cross-entropy loss')
plt.show()


# %%
model.eval()
for idx, sample in enumerate(test_loader):
    out = model(sample['obs'])
    loss = criterion(out,sample['label'])
    print(loss.ite)
    break


# %%
test_loss = []
for idx, sample in tqdm(enumerate(test_loader)):
    out = model(sample['obs'])
    loss = criterion(out,sample['label'])
    test_loss.append(loss.item())
print(np.mean(np.array(test_loss)))

