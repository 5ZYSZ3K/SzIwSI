import os
import torch
from model import EEGSignalDataset, NNet
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np


MODEL_PATH = "model.pt"
FILE_PATH = './grasp-and-lift-eeg-detection'

def read_csv(data, events):
    x = pd.read_csv(data)
    y = pd.read_csv(events)
    x = x.iloc[:,1:].values
    y = y.iloc[:,1:].values
    return x, y
    
trainset = []
events = []
data_file_name = os.path.join(f'./{FILE_PATH}/train/subj1_series1_data.csv')
events_file_name = os.path.join(f'./{FILE_PATH}/train/subj1_series1_events.csv')
x_test, y_test = read_csv(data_file_name, events_file_name)

dataset = EEGSignalDataset([x_test.T.astype(np.float32)], [y_test.T.astype(np.float32)], soft_label=False)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = NNet()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
dataiter = iter(dataloader)
x_test, y_test = next(dataiter)
outputs = model(x_test)
for i, output in enumerate(outputs):
    output_thresholded = (output > 0.5).float()
    print(f"Output {i}: {float(output_thresholded[0])} {float(output_thresholded[1])} {float(output_thresholded[2])} {float(output_thresholded[3])} {float(output_thresholded[4])} {float(output_thresholded[5])}")
    print(f"Target {i}: {y_test[i][0]} {y_test[i][1]} {y_test[i][2]} {y_test[i][3]} {y_test[i][4]} {y_test[i][5]}\n")