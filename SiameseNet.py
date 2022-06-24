"""
SiameseNet.py

Siamese network trained to learn differences between neural data
Neural data is collected with a Utah array in the motor cortex of a subject 
doing a 2D center-out task with 8 targets

"""

import random
import numpy as np
import pandas as pd
import os
import time
import torch
import torchvision
from torch import nn



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
class Triplet(torch.utils.data.Dataset):
    
    # Simple init
    def __init__(self, exppath, device):
        self.exppath = exppath
        self.device = device
        
        self.getTriplets()
           
    # Get file paths of all triplets. All possible anchor and positive combinations with randomly chosen negatives         
    def getTriplets(self):
        self.alabel = []    # List containing paths to the anchor file
        self.plabel = []    # List containing paths to the positive file
        self.nlabel = []    # List containing paths to the negative file
        self.atype = []     # Trialtype of anchor. Positive should be the same.
        self.ntype = []     # Trialtype of negative
                    
        trialpaths = list(os.listdir(self.exppath))
        
        for trialtype in trialpaths:
            
            trialpath = os.path.join(exppath, trialtype)
            trials = list(os.listdir(trialpath))
            num_trials = len(trials)
            
            for i in range(num_trials):
                for j in range(i+1, num_trials):
                    self.alabel.append(os.path.join(trialpath, trials[i]))
                    self.plabel.append(os.path.join(trialpath, trials[j]))
                    self.atype.append(trialtype)
                    
                    n_trialtype = random.choice(trialpaths)
                    while n_trialtype == trialtype:
                        n_trialtype = random.choice(trialpaths)
                    self.ntype.append(n_trialtype)
                    
                    n_trialpath = os.path.join(exppath, n_trialtype)
                    n_files = list(os.listdir(n_trialpath))
                    
                    self.nlabel.append(os.path.join(n_trialpath, random.choice(n_files)))
                
    # Length of dataset
    def __len__(self):
        return len(self.alabel)                        
        
    # Load and return the data for each triplet
    def __getitem__(self, idx):
        anchor = self.loadData(self.alabel[idx])
        positive = self.loadData(self.plabel[idx])
        negative = self.loadData(self.nlabel[idx])
        return anchor, positive, negative, idx, # self.atype[idx], self.ntype[idx] # For debugging
    
    # Load data, all saved data should be parquet files
    def loadData(self, filepath):
        data = pd.read_parquet(filepath)
        data = pd.DataFrame.to_numpy(data)
        data = torch.from_numpy(data)
        data = data.to(self.device)
        return data
    

# Set up path to dataset
exppath = os.getcwd() + '/Data/Spanky-170901-131421'
data = Triplet(exppath, device)

# Split dataset into train and test data
testsize = int(0.2*np.floor(len(data)))
trainsize = len(data) - testsize
train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])

# Define dataloaders
batch_size = 32     
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


''' Define model '''
class siamese(nn.Module):                    
    def __init__(self, bidirectional): 
        super(siamese,self).__init__()    
        
        # Convolutional layers
        kernel_size = 3         
        padding = int((kernel_size-1)/2)   # Keep same length with stride 1
        self.Conv1 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=kernel_size, padding=padding)
        
        lin = 1500  # input length is set to 1500
        lout = 150  # desired output length. Too long and it will slow down the LSTM
        kernel_size = lin-lout+1
        self.MaxPool1 =  nn.MaxPool1d(kernel_size)
        
        kernel_size = 3         
        padding = int((kernel_size-1)/2)   # Keep same length with stride 1
        self.Conv2 = nn.Conv1d(in_channels=32, out_channels=8, kernel_size=kernel_size, padding=padding)
        
        lin = lout
        seq_len = 15
        kernel_size = lin-seq_len+1
        self.MNaxPool2 = nn.MaxPool2d(kernel_size)        
        
        # LSTM
        self.hidden_size = 10           # Number of hidden features
        self.num_layers = 1             # Number of stacks
        self.D = 1                      # Constant for bidirectional LSTMs
        if bidirectional:
            self.D = 2
            
        # LSTM requires input size (number of features), the sequence length, and whether it's bidirectional
        self.LSTM = nn.LSTM(8, self.hidden_size, self.num_layers, bidirectional=bidirectional, batch_first=True)

        # Rather than 1 output, have as many as the sequence length (many-to-many)
        self.Linear = nn.Linear(self.D * self.hidden_size * seq_len, seq_len) 
        
        
    # Have to reinitialize h0 and c0 each pass as the shape depends on the batch size
    def h0c0(self, batch_size):
        h0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.hidden = (h0, c0)

    # Forward propagation
    def forward(self, x):      
        
        # Go through convolutional layers
        conv = self.Conv1(x)
        conv = self.MaxPool1(conv)
        conv = self.Conv2(conv)
        conv = self.MaxPool2(conv)
        
        # Initialize hidden and cell states
        batch_size = x.shape[0] 
        self.h0c0(batch_size)
        
        # LSTM layer
        lstm, self.hidden = self.LSTM(x, self.hidden)
        
        # Fully connected layer
        fc = torch.flatten(lstm, start_dim=1)
        y = self.Linear(fc)
        
        return y, #lstm, conv # For debugging


def distance(a, b):
    return np.sum(np.square(a-b), axis=-1)


# Initialize model
model = siamese(False)
model = model.float()
model = model.to(device)





















