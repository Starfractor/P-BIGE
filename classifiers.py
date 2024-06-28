import numpy as np
import os 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import torch
import torch.nn as nn
import random

max_seq_len = 64
batch_size = 512
stride = 3


# input_size = max_seq_len * 299 #36 #263
input_size = max_seq_len * 263 #263
hidden_size1 = 256
hidden_size2 = 256
num_classes = 13 #len(np.unique(Y))  # Assuming Y contains integer labels
learning_rate = 0.0005
num_epochs = 100



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, mcs_classes=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3_lab = nn.Linear(hidden_size1, num_classes)
        # self.fc3_mcs = nn.Linear(hidden_size1,mcs_classes)
        # self.dp = nn.Dropout(0)
        self.relu = nn.ReLU()

    def forward(self, x, return_embeddings = False):
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x)) 
        if return_embeddings:
            return x
        # x = self.relu(self.fc2(x)) + x
        x_lab = self.fc3_lab(x)
        # x_mcs = self.fc3_mcs(x)
        return x_lab

def get_classifier(weight_path=None):
    model = MLP(input_size, hidden_size1, hidden_size2, num_classes) 
    
    if weight_path is not None: 
        model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    model.eval()

    return model 