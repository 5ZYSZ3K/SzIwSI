import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import gc

FILE_PATH = './grasp-and-lift-eeg-detection'
list_dir = os.listdir(FILE_PATH)

labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff',
       'Replace', 'BothReleased']

torch.manual_seed(2021)
np.random.seed(2021)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

def read_csv(data, events):
    x = pd.read_csv(data)
    y = pd.read_csv(events)
    x = x.iloc[:,1:].values
    y = y.iloc[:,1:].values
    return x, y
    
trainset = []
gt = []
for filename in os.listdir(f'./{FILE_PATH}/train'):
    if 'data' in filename:
        data_file_name = os.path.join(f'./{FILE_PATH}/train', filename)
        id = filename.split('.')[0]
        events_file_name = os.path.join(f'./{FILE_PATH}/train', '_'.join(id.split('_')[:-1]) + '_events.csv')
        x, y = read_csv(data_file_name, events_file_name)
        trainset.append(x.T.astype(np.float32))
        gt.append(y.T.astype(np.float32))

valid_dataset = trainset[-1:]
valid_gt = gt[-1:]
trainset = trainset[:-1]
gt = gt[:-1]

all_data = np.concatenate(trainset, axis=1)  # trainset: [(32, N1), (32, N2), ...]
m = np.mean(all_data, axis=1, keepdims=True)
s = np.std(all_data, axis=1, keepdims=True)

trainset = [(data - m) / s for data in trainset]

valid_dataset = [(data - m) / s for data in valid_dataset]

chunk_size = 1000

def resample_data(gt):
    """
    split long signals to smaller chunks, discard no-events chunks  
    """
    total_discard_chunks = 0
    mean_val = []
    threshold = 0.01
    index = []
    
    for i in range(len(gt)):
        for j in range(0, gt[i].shape[1], chunk_size):
            mean_val.append(np.mean(gt[i][:, j:min(gt[i].shape[1],j+chunk_size)]))
            if mean_val[-1] < threshold:
                total_discard_chunks += 1
            else:
                index.extend([(i, k) for k in range(j, min(gt[i].shape[1],j+chunk_size))])

    # plt.plot([0, len(mean_val)], [threshold, threshold], color='r')
    # plt.scatter(range(len(mean_val)), mean_val, s=1)
    # plt.show()
    print('Total number of chunks discarded: {} chunks'.format(total_discard_chunks))
    print('{}% data'.format(total_discard_chunks/len(mean_val)))
    del mean_val
    gc.collect()
    return index

class EEGSignalDataset(Dataset):
    def __init__(self, data, gt, m=m, s=s, soft_label=True, train=True):
        self.data = data
        self.gt = gt
        self.train = train
        self.soft_label = soft_label
        self.eps = 1e-7
        if train:
            self.index = resample_data(gt)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
        for dt in self.data:
            dt -= m
            dt /= s + self.eps
    
    def __getitem__(self, i):
        i, j = self.index[i]
        raw_data, label = self.data[i][:, max(0, j - 1024 + 1):j + 1], self.gt[i][:, j]

        pad = 1024 - raw_data.shape[1]
        if pad:
            raw_data = np.pad(raw_data, ((0, 0), (pad, 0)), 'constant', constant_values=0)

        raw_data = torch.from_numpy(raw_data.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        if self.soft_label:
            label[label < .02] = .02

        return raw_data, label

    def __len__(self):
        return len(self.index)

dataset = EEGSignalDataset(trainset, gt) 
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
print(len(dataset))

class NNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=6):
        super(NNet, self).__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 5, padding=2),
            nn.Conv1d(self.hidden, self.hidden, 16, stride=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden, self.hidden, 7, padding=3),
        )
        for i in range(6):
            self.net.add_module('conv{}'.format(i), \
                                self.__block(self.hidden, self.hidden))
        self.net.add_module('final', nn.Sequential(
            nn.Conv1d(self.hidden, out_channels, 1),
            nn.Sigmoid()
        ))
        
    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
        )
    
    def forward(self, x):
        return self.net(x)
    
n_epochs = 10
    
nnet = NNet()
nnet.to(device)
loss_fnc = nn.BCELoss()
adam = optim.Adam(nnet.parameters(), lr=0.002, betas=(0.5, 0.99))
loss_his, train_loss = [], []
nnet.train()
for epoch in range(n_epochs):
    p_bar = dataloader
    print(len(p_bar))
    for i, (x, y) in enumerate(p_bar):
        x, y = x.to(device), y.to(device)
        pred = nnet(x)
        loss = loss_fnc(pred.squeeze(dim=-1), y)
        adam.zero_grad()
        loss.backward()
        adam.step()
        train_loss.append(loss.item())
        # p_bar.set_description('[Loss: {}]'.format(train_loss[-1]))
        if i % 50 == 0:
            loss_his.append(np.mean(train_loss))
            train_loss.clear()
    print('[Epoch {}/{}] [Loss: {}]'.format(epoch+1, n_epochs, loss_his[-1]))
    
torch.save(nnet.state_dict(), 'model.pt')