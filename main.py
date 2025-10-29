import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import gc

from model import EEGSignalDataset, NNet

FILE_PATH = './grasp-and-lift-eeg-detection'
list_dir = os.listdir(FILE_PATH)

limit = 128


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
events = []
for filename in os.listdir(f'./{FILE_PATH}/train'):
    if 'data' in filename:
        data_file_name = os.path.join(f'./{FILE_PATH}/train', filename)
        id = filename.split('.')[0]
        events_file_name = os.path.join(f'./{FILE_PATH}/train', '_'.join(id.split('_')[:-1]) + '_events.csv')
        x_train, y_train = read_csv(data_file_name, events_file_name)
        trainset.append(x_train.T.astype(np.float32)[:, :limit])  # we limit the intake
        events.append(y_train.T.astype(np.float32)[:, :limit])  # we limit the intake

valid_dataset = trainset[-1:]
valid_events = events[-1:]
trainset = trainset[:-1]
events = events[:-1]

all_data = np.concatenate(trainset, axis=1)  # trainset: [(32, N1), (32, N2), ...]
m = np.mean(all_data, axis=1, keepdims=True)
s = np.std(all_data, axis=1, keepdims=True)
trainset = [(data - m) / s for data in trainset]
valid_dataset = [(data - m) / s for data in valid_dataset]

dataset = EEGSignalDataset(trainset, events) # TODO run it with soft_labels and check if it's better
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
print(len(dataset))
    
n_epochs = 10
    
model = NNet()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.5, 0.99))
loss_his, train_loss = [], []
loss_history, train_loss = [], []
model.train()

for epoch in range(n_epochs):
    print('[Epoch {}/{}] started'.format(epoch+1, n_epochs))
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