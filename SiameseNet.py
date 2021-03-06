"""
SiameseNet.py

Siamese network trained to learn differences between neural data
Neural data is collected with a Utah array in the motor cortex of a subject 
doing a 2D center-out task with 8 targets

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
class TripletData(torch.utils.data.Dataset):
    
    # Simple init
    def __init__(self, exppath, device):
        self.exppath = exppath
        self.device = device
        
        self.data = {}
        trialpaths = list(os.listdir(self.exppath))
        for trialtype in trialpaths:   
            trialpath = os.path.join(exppath, trialtype)
            trials = list(os.listdir(trialpath))
            for t in trials:
                self.data[(trialtype, t)] = self.loadData(os.path.join(trialpath, t))
        
        self.getTriplets()
     
    # Load data, all saved data should be parquet files
    # Can change it here to get spectra instead of raw LFP
    def loadData(self, filepath):
        data = pd.read_parquet(filepath)
        data = pd.DataFrame.to_numpy(data)
        data = torch.from_numpy(data)
        data = data.float()
        return data    
     
    # Set trial type and trial number for each triplet
    def getTriplets(self):
        self.alabel = []    # List containing paths to the anchor file
        self.plabel = []    # List containing paths to the positive file
        self.nlabel = []    # List containing paths to the negative file
        self.atype = []     # Trialtype of anchor. Positive should be the same.
        self.ntype = []     # Trialtype of negative
                    
        trialpaths = list(os.listdir(self.exppath))
        
        # Loop through each trial type. For each possible anchor-positive pair, find a random negative 
        for trialtype in trialpaths:
            
            trialpath = os.path.join(self.exppath, trialtype)
            trials = list(os.listdir(trialpath))
            num_trials = len(trials)
            
            for i in range(num_trials):
                for j in range(i+1, num_trials):
                    self.alabel.append((trialtype, trials[i]))
                    self.plabel.append((trialtype, trials[j]))
                    self.atype.append(trialtype)
                    
                    n_trialtype = random.choice(trialpaths)
                    while n_trialtype == trialtype:
                        n_trialtype = random.choice(trialpaths)
                    self.ntype.append(n_trialtype)
                    
                    n_trialpath = os.path.join(exppath, n_trialtype)
                    n_files = list(os.listdir(n_trialpath))
                    
                    self.nlabel.append((n_trialtype, random.choice(n_files)))
                
    # Length of dataset
    def __len__(self):
        return len(self.alabel)                        
        
    # Return the data for each index
    def __getitem__(self, idx):
        anchor = self.data[self.alabel[idx]].to(self.device)
        positive = self.data[self.plabel[idx]].to(self.device)
        negative = self.data[self.nlabel[idx]].to(self.device)
        return anchor, positive, negative, idx, # self.atype[idx], self.ntype[idx] # For debugging
    
   

# Set up path to dataset
exppath = os.getcwd() + '/Data/Spanky-170901-131421'
data = TripletData(exppath, device)

# Split dataset into train and test data
testsize = int(0.2*np.floor(len(data)))
trainsize = len(data) - testsize
train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])

# Define dataloaders
batch_size = 64    # Larger batch sizes better for triplet loss since we want semi-hard triplets
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


''' Define model '''
class siamese(nn.Module):                    
    def __init__(self, bidirectional, device): 
        super(siamese,self).__init__()    
        
        self.device = device
        
        # Convolutional layers
        kernel_size = 3         
        padding = int((kernel_size-1)/2)   # Keep same length with stride 1
        self.Conv1 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=kernel_size, padding=padding)
        
        lin = 1500  # input length is set to 1500
        lout = 150  # desired output length. Too long and it will slow down the LSTM
        kernel_size = int(lin / lout)
        self.MaxPool1 =  nn.MaxPool1d(kernel_size=kernel_size)
        
        # kernel_size = 3         
        # padding = int((kernel_size-1)/2)   # Keep same length with stride 1
        # self.Conv2 = nn.Conv1d(in_channels=32, out_channels=8, kernel_size=kernel_size, padding=padding)
        
        # lin = lout
        # seq_len = 15
        # kernel_size = int(lin / seq_len)
        # self.MaxPool2 = nn.MaxPool1d(kernel_size=kernel_size)     
        seq_len = lout
        
        # LSTM
        self.hidden_size = 10           # Number of hidden features
        self.num_layers = 1             # Number of stacks
        self.D = 1                      # Constant for bidirectional LSTMs
        if bidirectional:
            self.D = 2
            
        # LSTM requires input size (number of features), the sequence length, and whether it's bidirectional
        self.LSTM = nn.LSTM(32, self.hidden_size, self.num_layers, bidirectional=bidirectional, batch_first=True)

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
        conv1 = self.Conv1(x)
        conv1 = self.MaxPool1(conv1)
        # conv2 = self.Conv2(conv1)
        # conv2 = self.MaxPool2(conv2)
        
        conv1 = conv1.permute(0, 2, 1)  # Need to reorder dimensions for LSTM
        
        # Initialize hidden and cell states
        batch_size = x.shape[0] 
        self.h0c0(batch_size)
        
        # LSTM layer
        lstm, self.hidden = self.LSTM(conv1, self.hidden)
        
        # Fully connected layer
        fc = torch.flatten(lstm, start_dim=1)
        y = self.Linear(fc)
        
        # Keep length to 1 (project to unit circle)
        y = nn.functional.normalize(input=y, p=2, dim=1) 
        
        return y #, lstm, conv1, conv2 # For debugging


# For calculating distance to find semi-hard triplets
def distance(a, b):
    dist = (a-b).pow(2)
    dist = dist.sum(1).sqrt()
    return dist


# Initialize model
model = siamese(False, device)
model = model.float()
model = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)

# Loss function
alpha = 0.2
triplet_loss = nn.TripletMarginLoss(margin=alpha)

epochs = 100
verbose_steps = 20
TrainLoss = np.zeros((epochs, len(trainloader)))
start = time.time()
for e in range(epochs):
    
    data.getTriplets()
    train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    b = 0
    for anchor, positive, negative, idx in trainloader:
    
        # Find semi-hard triplets to train from
        model.eval()
        
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        positive_dist = distance(anchor_out, positive_out)
        negative_dist = distance(anchor_out, negative_out)
        
        easy = positive_dist + alpha < negative_dist
        hard = negative_dist < positive_dist
        semi_hard = (positive_dist < negative_dist) & (negative_dist < positive_dist+alpha)
        
        if e != 0 and np.mean(TrainLoss[e-1, :]) < 0.15:
            train_hard = True
        
        keep = semi_hard
        
        # Restructure to train on semi-hard triplets
        anchor = anchor[keep, :, :]
        positive = positive[keep, :, :]
        negative = negative[keep, :, :]
        
        model.train()
        
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = triplet_loss(anchor_out, positive_out, negative_out)
        
        # Gradient step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


## Save trained model
modelfile = os.getcwd() + '/Models/Initial_Test_shuffle.pt'
torch.save(model.state_dict(), modelfile)
print('Model Saved')


## Save losses
lossfile = os.getcwd() + '/Losses/Initial_Test_shuffle.pkl'
file = open(lossfile,'wb')
pickle.dump(TrainLoss, file)
print('Losses Saved')
file.close()



# Test triplets 
anchor, positive, negative, idx = next(iter(trainloader))
anchor_out = model(anchor)
positive_out = model(positive)
negative_out = model(negative)

positive_dist = distance(anchor_out, positive_out)
negative_dist = distance(anchor_out, negative_out)

plt.hist(positive_dist.cpu().detach())
plt.hist(negative_dist.cpu().detach())

