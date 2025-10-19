import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
x, y = [], []

TEST_SUBJECT_1_SERIES_9_PATH = './grasp-and-lift-eeg-detection/test/subj1_series9_data.csv'
TRAIN_SUBJECT_1_SERIES_1_DATA_PATH = './grasp-and-lift-eeg-detection/train/subj1_series1_data.csv'
TRAIN_SUBJECT_1_SERIES_1_EVENTS_PATH = './grasp-and-lift-eeg-detection/train/subj1_series1_data.csv'

x = pd.read_csv(TRAIN_SUBJECT_1_SERIES_1_DATA_PATH).iloc[:, 1:].values.astype(np.float32)
y = pd.read_csv(TRAIN_SUBJECT_1_SERIES_1_EVENTS_PATH).iloc[:, 1:].values.astype(np.float32)
x_train, x_test = train_test_split(x, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)
