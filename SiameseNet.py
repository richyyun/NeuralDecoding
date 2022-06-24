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
import torch
import time
import torchvision


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
        return anchor, positive, negative, # self.atype[idx], self.ntype[idx] # For debugging
    
    # Load data, all saved data should be parquet files
    def loadData(self, filepath):
        data = pd.read_parquet(filepath)
        return pd.DataFrame.to_numpy(data)
    

# Set up path to dataset
exppath = os.getcwd() + '/Data/Spanky-170901-131421'
data = Triplet(exppath, device)











