import torch
import torch.nn as nn
import torch.optim as optim
import os, re, glob, shutil, random
import pandas as pd
import numpy as np
import csv_devider
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def csvpath2tensor(path):
    csv_df = pd.read_csv(path)
    csv_list = csv_df.values.T.tolist()
    csv_list = np.delete(csv_list, [0,1,2,3], 0)
    adjusted_csv_list = []
    neck_mean = (np.mean(csv_list[0]) + np.mean(csv_list[1]))/2
    for list in csv_list:
        l_i = [int(s) for s in list]
        list = [pd.np.nan if i == -1 else i for i in l_i]
        adjusted_series = pd.Series(list)
        adjusted_series = adjusted_series.interpolate(limit_direction='forward')
        adjusted_list = adjusted_series.values.tolist()
        remove_nan = np.nan_to_num(adjusted_list, nan=neck_mean)
        adjusted_csv_list.append(remove_nan)
    adjusted_csv_list_t = np.array(adjusted_csv_list).T.tolist()
    d_from_center_t = []
    for i, frame in enumerate(adjusted_csv_list_t):
        neck_x = frame[0]
        neck_y= frame[1]
        R_shoulder_x = abs(frame[2] - neck_x)
        R_shoulder_y = abs(frame[3] - neck_y)
        R_elbow_x = abs(frame[4] - neck_x)
        R_elbow_y = abs(frame[5] - neck_y)
        R_wrist_x = abs(frame[6] - neck_x)
        R_wrist_y = abs(frame[7] - neck_y)
        L_shoulder_x = abs(frame[8] - neck_x)
        L_shoulder_y = abs(frame[9] - neck_y)
        L_elbow_x = abs(frame[10] - neck_x)
        L_elbow_y = abs(frame[11] - neck_y)
        L_wrist_x = abs(frame[12] - neck_x)
        L_wrist_y = abs(frame[13] - neck_y)
        d_from_center_elem = [R_shoulder_x, R_shoulder_y, R_elbow_x, R_elbow_y, R_wrist_x, R_wrist_y, L_shoulder_x, L_shoulder_y, L_elbow_x, L_elbow_y, L_wrist_x, L_wrist_y]
        d_from_center_t.append(d_from_center_elem)
    d_from_center = np.array(d_from_center_t).T.tolist()
    normalized_d_from_center = []
    for keys in d_from_center:
        key_min = min(keys)
        key_max = max(keys)
        if (key_min == key_max):
            diff = 1
        else:
            diff = key_max - key_min
        normalized_keys = [round((i - key_min) / diff,4) for i in keys]
        normalized_d_from_center.append(normalized_keys)
    return torch.tensor(normalized_d_from_center)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, tagset_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, lstm_out = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
        tag_scores = self.softmax(tag_space)
        return tag_scores

width_of_csv = csv_devider.width_of_csv
INPUT_SIZE = width_of_csv
HIDDEN_DIM = 128
TAG_SIZE = 9

model = LSTMClassifier(INPUT_SIZE, HIDDEN_DIM, TAG_SIZE).to(device)
model.load_state_dict(torch.load('weights/lr0001ep12.pth'))
best_model = model
path = 'csv/devided_csv/WIN_20211129_12_06_26_Pro.mp4.csv/*.csv'
testdatas = glob.glob(path)
predict_list = []
for testdata in testdatas:
    inputs = csvpath2tensor(testdata).view(12, 1, INPUT_SIZE)
    inputs = inputs.to(device, dtype=torch.float)
    out = best_model(inputs)
    out = out.tolist()
    predict_list.append(out)

predicted_motion = []
for predict in predict_list:
    if max(predict[0]) <= -0.1:
        predicted_motion.append(str('-'))
    else:
        predicted_motion.append(str(predict[0].index(max(predict[0]))))

print(predicted_motion)
with open('motionpredictionresult/WIN_20211129_12_06_26_Pro_010.txt', 'w') as f:
    f.writelines(predicted_motion)