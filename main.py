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


# this is to ensure reproductability of our code. 
torch.manual_seed(2025)
np.random.seed(2025)

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
        x_train, y_train = read_csv(data_file_name, events_file_name)
        trainset.append(x_train.T.astype(np.float32))
        gt.append(y_train.T.astype(np.float32))

valid_dataset = trainset[-1:]
valid_gt = gt[-1:]
trainset = trainset[:-1]
gt = gt[:-1]

# normalize data
all_data = np.concatenate(trainset, axis=1)  # trainset: [(32, N1), (32, N2), ...]
m = np.mean(all_data, axis=1, keepdims=True)
s = np.std(all_data, axis=1, keepdims=True)
trainset = [(data - m) / s for data in trainset]
valid_dataset = [(data - m) / s for data in valid_dataset]

chunk_size = 1000

# split long signals to smaller chunks, discard no-events chunks 
# if chunk's mean is lower than threshold we discard it, because there was no activity during that time
# we keep a record of indices that remained in training dataset
def clean_and_resample_data(gt):
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
            self.index = clean_and_resample_data(gt)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
        for dt in self.data:
            dt -= m
            dt /= s + self.eps # TODO it seems like data is normalized again here? should we remove it?
    
    # get chunk of signal, if chunk is smaller than 1024 we add 0 padding
    # if soft_label we avoid using exact zeros - we use 0.2 instead
    # this is for loss functions with logarithms
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

dataset = EEGSignalDataset(trainset, gt) # TODO run it with soft_labels and check if it's better
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
print(len(dataset))

# we return 6 channels because our dataset specify 6 hand gestures
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
        for i in range(6): # six different hand gestures
            self.net.add_module('conv{}'.format(i), \
                                self.__block(self.hidden, self.hidden))
        self.net.add_module('final', nn.Sequential(
            nn.Conv1d(self.hidden, out_channels, 1),
            nn.Sigmoid() # produces 1 or 0, depending on input
        ))
        
    def __block(self, inchannels, outchannels):
        return nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Dropout(p=0.1, inplace=True),
            nn.Conv1d(inchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(outchannels, outchannels, 5, padding=2),
            nn.LeakyReLU(0.1), # TODO maybe we should try PReLU? - The learnable slope adjusts over time, which can yield gradients that are closer to optimal, especially in deeper networks.
        )
    
    def forward(self, x):
        return self.net(x)
    
n_epochs = 10
    
model = NNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
loss_history, train_loss = [], []
model.train()

for epoch in range(n_epochs):
    for i, (x_train, y_train) in enumerate(dataloader):
        predicted_y = model(x_train)
        loss = loss_function(predicted_y.squeeze(dim=-1), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if i % 50 == 0:
            loss_history.append(np.mean(train_loss))
            train_loss.clear()
    print('[Epoch {}/{}] [Loss: {}]'.format(epoch+1, n_epochs, loss_history[-1]))
    
torch.save(model.state_dict(), 'model.pt')