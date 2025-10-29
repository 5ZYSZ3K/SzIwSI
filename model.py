import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

chunk_size = 1000

# split long signals to smaller chunks, discard no-events chunks 
# if chunk's mean is lower than threshold we discard it, because there was no activity during that time
# we keep a record of indices that remained in training dataset
def clean_and_resample_data(data):
    index = []

    for i in range(len(data)):
        for j in range(0, data[i].shape[1], chunk_size):
            index.extend([(i, k) for k in range(j, min(data[i].shape[1],j+chunk_size))])

    # plt.plot([0, len(mean_val)], [threshold, threshold], color='r')
    # plt.scatter(range(len(mean_val)), mean_val, s=1)
    # plt.show()
    return index


class EEGSignalDataset(Dataset):
    def __init__(
            self, 
            data, 
            events, 
            soft_label=True, 
            train=True
        ):
        self.data = data
        self.events = events
        self.train = train
        self.soft_label = soft_label
        if train:
            self.index = clean_and_resample_data(events)
        else:
            self.index = [(i, j) for i in range(len(data)) for j in range(data[i].shape[1])]
        # for i, gt in enumerate(self.data):
        #     gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        #     self.data[i] = gt

    # get chunk of signal, if chunk is smaller than 1024 we add 0 padding
    # if soft_label we avoid using exact zeros - we use 0.2 instead
    # this is for loss functions with logarithms
    def __getitem__(self, i):
        i, j = self.index[i]
        raw_data, label = self.data[i][:, max(0, j - 1024 + 1):j + 1], self.events[i][:, j]

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

# we return 6 channels because our dataset specify 6 hand gestures
class NNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=6):
        super(NNet, self).__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2),
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