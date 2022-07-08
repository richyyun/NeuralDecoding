"""
DataNormalization.py

Normalizes for each trial across each channel
Plots the averaged LFP trace of each channel for each trial type 

"""
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
plt.ioff()

datapath = os.getcwd() + '/Data'

experiments = list(os.listdir(datapath))

fs = 500
win = np.arange(np.round(-0.5*fs), np.round(2.5*fs))

# Save a 2D numpy array into a parquet file using pandas
def saveParquet(data, savefile):
    df = pd.DataFrame(data)
    df.columns = df.columns.astype(str)
    df.to_parquet(savefile)

# Calculate moving average 
def movmean(data, win):
    for i in range(data.shape[0]):
        data[i,:] = np.convolve(data[i,:], np.ones((win)), mode='same')/win
    return data

for exp in experiments:
    print('Starting ' + exp + ', ' + str(experiments.index(exp)+1) + '/' + str(len(experiments)))    
    startexp = time.time()
    
    exppath = os.path.join(datapath, exp)
    trialtypes = list(os.listdir(exppath))
    
    
    fig = plt.figure() 
    fig.set_size_inches(10,10)
    for trial in trialtypes:
        if(not str.isdigit(trial)):
            continue
        print('Trial ' + trial + '...', end ="")
        start = time.time()
        
        trialpath = os.path.join(exppath, trial)
        trialnums = list(os.listdir(trialpath))
        
        avgtrial = []
        for t in trialnums:
            if 'norm' in t:
                continue;
            
            # Load
            file = os.path.join(trialpath, t)
            data = pd.read_parquet(file)
            data = pd.DataFrame.to_numpy(data)
            
            # Normalize over time per channel - need to keep relative changes alive
            data = (data-np.mean(data, axis=1, keepdims=True))/np.std(data, axis=1, keepdims=True)
            
            # Save
            savefile = trialpath + '/' + str.split(t,'.')[0] + '_norm.parquet'
            saveParquet(data, savefile)            
            
            # Add to cumulative average
            if len(avgtrial) == 0:
                avgtrial = data/len(trialnums)
            else:
                avgtrial += data/len(trialnums)
        
        savefile = trialpath + '/Averaged_norm.parquet'
        saveParquet(avgtrial, savefile)
        
        # Plot smoothed average 
        smoothed = movmean(avgtrial, 25)
        
        ax = plt.subplot(3, 3, int(trial))
        plt.plot(win/fs, smoothed.T, zorder=1)
        yl = plt.ylim()
        plt.vlines(0, yl[0], yl[1], zorder=2)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_ticks([-0.5,0,2.5])
        ax.get_yaxis().set_ticks([yl[0], 0, yl[1]])
        
        end = time.time()
        print('Done, ' + str(round(end-start, 2)) + 's')
    
    figname = exppath + '/AllTrials.eps'
    plt.savefig(figname, format='eps')
    plt.close(fig)
                    
    end = time.time()
    print('Total, ' + str(round(end-startexp, 2)) + 's')
    
    
    
    