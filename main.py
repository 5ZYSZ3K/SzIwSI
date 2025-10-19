import numpy as np
from TorchModel import TorchModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
import torch.optim as optim
import torch.nn as nn
import torch

x, y = [], []

TEST_SUBJECT_1_SERIES_9_PATH = './grasp-and-lift-eeg-detection/test/subj1_series9_data.csv'
TRAIN_SUBJECT_1_SERIES_1_DATA_PATH = './grasp-and-lift-eeg-detection/train/subj1_series1_data.csv'
TRAIN_SUBJECT_1_SERIES_1_EVENTS_PATH = './grasp-and-lift-eeg-detection/train/subj1_series1_data.csv'

x = pd.read_csv(TRAIN_SUBJECT_1_SERIES_1_DATA_PATH).iloc[:, 1:].values.astype(np.float32)
y = pd.read_csv(TRAIN_SUBJECT_1_SERIES_1_EVENTS_PATH).iloc[:, 1:].values.astype(np.float32)
x_train, x_test = train_test_split(x, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)

net = TorchModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 4
trainloader = torch.utils.data.DataLoader(x_train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(x_test, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # pobranie danych
        inputs, labels = data

        # ustawienie wszystkich optymalizowanych gradientow na zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # wydrukowanie statystyk
        running_loss += loss.item()
        if i % 2000 == 1999:    # wydrukowanie wynikow co 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Trening ukończony')