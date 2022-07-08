# -*- coding: utf-8 -*-
"""
RNN.py

Recurrent neural network to predict the type of trial using the normalized data
"""

import random
import numpy as np
import pandas as pd
import pickle
import os
import time
import torch
from torch import nn
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
class Data(torch.utils.data.Dataset):
    
    # Simple init
    def __init__(self, exppath, device):
        self.exppath = exppath
        self.device = device
        
        self.data = []
        self.label = []
        trialpaths = list(os.listdir(self.exppath))
        # Load all data and trial type label
        for trialtype in trialpaths:   
            if(not str.isdigit(trialtype)):
                continue
            trialpath = os.path.join(exppath, trialtype)
            trials = list(os.listdir(trialpath))
            for t in trials:
                if 'norm' in t:
                    self.data.append(self.loadData(os.path.join(trialpath, t)))
                    self.label.append(trialtype)
     
    # Load data, all saved data should be parquet files
    # Can change it here to get spectra instead of raw LFP
    def loadData(self, filepath):
        data = pd.read_parquet(filepath)
        data = pd.DataFrame.to_numpy(data)
        data = torch.from_numpy(data)
        data = data.float()
        return data    
     
    # Length of dataset
    def __len__(self):
        return len(self.data)                        
        
    # Return the data for each index
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
# Dataset
exppath = os.getcwd() + '/Data/Spanky-170901-131421'
data = Data(exppath, device)

# Split dataset into train and test data
testsize = int(0.2*np.floor(len(data)))
trainsize = len(data) - testsize
train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])

# Define dataloaders
batch_size = 8    
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


''' Define model '''
class RNN(nn.Module):                    
    def __init__(self, bidirectional, device): 
        super(RNN,self).__init__()    
        
        self.device = device
        
        # Convolutional layers
        kernel_size = 3         
        padding = int((kernel_size-1)/2)   # Keep same length with stride 1
        self.Conv1 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=kernel_size, padding=padding)
        
        lin = 1500  # input length is set to 1500
        lout = 150  # desired output length. Too long and it will slow down the LSTM
        kernel_size = int(lin / lout)
        self.MaxPool1 =  nn.MaxPool1d(kernel_size=kernel_size)
        
        seq_len = lout
        # seq_len = 1500
        
        # LSTM
        self.hidden_size = 10           # Number of hidden features
        self.num_layers = 1             # Number of stacks
        self.D = 1                      # Constant for bidirectional LSTMs
        if bidirectional:
            self.D = 2
            
        nfeatures = 32
        
        # LSTM requires input size (number of features), the sequence length, and whether it's bidirectional
        self.LSTM = nn.LSTM(nfeatures, self.hidden_size, self.num_layers, bidirectional=bidirectional, batch_first=True)

        # Fully connected layers to a final softmax layer
        self.Linear1 = nn.Linear(self.D * self.hidden_size * seq_len, seq_len) 
        self.Linear2 = nn.Linear(seq_len, 9)
        self.Softmax = nn.Softmax(dim=1)
        
        
    # Have to reinitialize h0 and c0 each pass as the shape depends on the batch size
    def h0c0(self, batch_size):
        h0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.hidden = (h0, c0)

    # Forward propagation
    def forward(self, x):      
        
        # Go through convolutional layer
        conv1 = self.Conv1(x)
        conv1 = self.MaxPool1(conv1)

        conv1 = conv1.permute(0, 2, 1)  # Need to reorder dimensions for LSTM
        
        # Initialize hidden and cell states for the LSTM
        batch_size = x.shape[0] 
        self.h0c0(batch_size)
        
        # LSTM layer
        lstm, self.hidden = self.LSTM(conv1, self.hidden)
        
        # Fully connected layers
        fc = torch.flatten(lstm, start_dim=1)
        y = self.Linear1(fc)
        y = self.Linear2(y)
        
        # Softmax
        out = self.Softmax(y) 
        
        return out #y, lstm, conv1, conv2 # For debugging


# Initialize model
Model = RNN(False, device)
Model = Model.float()
Model = Model.to(device)
Model.train()

# Optimizer
Optimizer = torch.optim.Adam(Model.parameters(), lr=1e-4, weight_decay=1e-6)

# Loss function
LossFn = nn.CrossEntropyLoss()

# Start training
epochs = 100
verbose_steps = 20
TrainLoss = np.zeros((epochs, len(trainloader)))
start = time.time()
for e in range(epochs):
        
    b = 0
    for data, label in trainloader:
        
        data = data.to(device)
        
        label = torch.from_numpy(np.asarray(label).astype(int)-1)
        label = label.to(device)
          
        out = Model(data)
        
        loss = LossFn(out, label.type(torch.long))
        
        # Gradient step 
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        # Save losses
        TrainLoss[e, b] = loss.item()
        if np.isnan(TrainLoss[e,b]):
            break

        # Print statistics
        if b%verbose_steps == 0:
            print('Epoch:', e+1 , '/', epochs, ' Batch:', b+1, '/', len(trainloader))
            print('Training Loss:', TrainLoss[e, b])
            print('Time:', time.time()-start)
            
        b += 1












