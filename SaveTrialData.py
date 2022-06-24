import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io
import scipy.signal
import pandas as pd
import os
import time
from scipy.interpolate import interp1d

''' Function for loading matlab files '''
def loadmatfile(fname):
    # If it's a v7.3 file
    try:
        f = h5py.File(fname,'r')
    # Else
    except:
        f = scipy.io.loadmat(fname)
    data = {}
    for k, v in f.items():
        data[k] = np.array(v)
    return data

def gettrialinds(trig, win, lim):
    trialinds = np.tile(win, [trig.shape[1],1])
    trialinds = trialinds.T + trig
    bad = (trialinds[:][0] < 0) | (trialinds[:][-1] >= lim)
    trialinds = trialinds.T
    trialinds = trialinds[np.logical_not(bad)]
    return trialinds

tankpath = 'R:/Yun/Spanky/CycleTriggered-170710-143939'

fdata = 'CleanData.mat'
data = loadmatfile(tankpath+'/'+fdata)
fnames = np.squeeze(data['List'])
Experiments = [f[0] for f in fnames]
Duration = np.squeeze(data['Duration'])

savebasepath = 'C:/Users/Richy Yun/Dropbox/Projects/NeuralDecoding/Data'
newfs = 500

for exp in range(len(Experiments)):

    print('Starting ' + Experiments[exp] + ', ' + str(exp+1) + '/' + str(len(Experiments)))    
    startexp = time.time()
    
    blockname = Experiments[exp]
    
    
    print('Loading Data...', end ="")
    start = time.time()
    
    data = loadmatfile(tankpath+'/'+blockname+'/'+'Task.mat')
    trialtype = data['trialtype']
    trialstart = data['trialstart']
    
    if len(np.unique(trialtype)) < 8:
        print('Not enough trial types')
        continue
    
    saveexppath = savebasepath + '/' + blockname
    if os.path.isdir(saveexppath):
        continue
    
    os.mkdir(saveexppath)
    
    LFP = []
    for i in range(1,97):
        data = loadmatfile(tankpath+'/'+blockname+'/'+'LFP'+str(i)+'.mat')
        LFP.append(*data['LFP'])
    LFP = np.array(LFP)
    
    data = loadmatfile(tankpath+'/'+blockname+'/'+'LFPfs.mat')
    fs = data['fs']
    
    data = loadmatfile(tankpath+'/'+blockname+'/'+'BxID.mat')
    bfs = data['bfs']
        
    end = time.time()
    print('Done, ' + str(round(end-start, 2)) + 's')
    
    
    print('Resampling...', end ="")
    start = time.time()

    nsamples = int(np.round(LFP.shape[1]/fs*newfs))
    newlocs = np.linspace(0.0, 1.0, nsamples, endpoint=False)
    prevlocs = np.linspace(0.0, 1.0, LFP.shape[1], endpoint=False)
    
    dwnsample = np.empty((96, nsamples))
    for i in range(96):
        f = interp1d(prevlocs, LFP[i, :], kind='cubic')
        dwnsample[i, :] = f(newlocs)
        # dwnsample[i, :] = np.interp(newlocs, prevlocs, LFP)
    
    end = time.time()
    print('Done, ' + str(round(end-start, 2)) + 's')
    
    
    print('Saving Data')
    win = np.arange(np.round(-0.5*newfs), np.round(2.5*newfs))
    
    for i in range(1,10):   
        
        if i == 5:
            continue
        
        print('Trial ' + str(i) + '...', end ="")
        start = time.time()
            
        savefilepath = saveexppath + '/' + str(i)
        if not os.path.isdir(savefilepath):
            os.mkdir(savefilepath)
        
        trig = np.round(trialstart[trialtype==i]/bfs*newfs)
        trialinds = gettrialinds(trig.astype(int), win.astype(int), len(dwnsample[0, :]))

        alltrials = []
        for c in range(96):
            trials = dwnsample[c][trialinds]
            if c == 0:
                alltrials = np.empty((96, trials.shape[0], trials.shape[1]))
            alltrials[c] = trials

        alltrials = np.moveaxis(alltrials, 1, 0)
        
        for t in range(alltrials.shape[0]):
            df = pd.DataFrame(alltrials[t, :, :])
            df.columns = df.columns.astype(str)
            # table = pa.Table.from_pandas(df)
            savefile = savefilepath + '/' + str(t) + '.parquet'
            # pq.write_table(table, savefile)
            df.to_parquet(savefile)

        end = time.time()
        print('Done, ' + str(round(end-start, 2)) + 's')
    
    
    endexp = time.time()
    print('Total ' + str(round(endexp-startexp, 2)) + 's')




# # # Testing 
# # ''' Trial triggered averages of LFPs and spectra '''
# TrigAvg = {}
# Spectra = {}
# for i in range(1,10):
#     if i == 5:
#         continue
    
#     trig = np.round(trialstart[trialtype==i]/bfs*newfs)
#     win = np.arange(np.round(-0.5*newfs), np.round(2.5*newfs))
#     trialinds = gettrialinds(trig.astype(int), win.astype(int), len(dwnsample[0,:]))
    
#     allavg = np.empty((96, trialinds.shape[1]))
#     allspect = []
#     for c in range(96):
#         trials = dwnsample[c][trialinds]
#         allavg[c] = np.mean(trials, axis=0)
#         f, t, spect = scipy.signal.stft(trials, fs=np.squeeze(newfs), nperseg=200, noverlap=190, window='hamming')
#         if c == 0:
#             allspect = np.empty((96, spect.shape[1], spect.shape[2]))
#         allspect[c] = np.mean(abs(spect), axis=0)
        
#     TrigAvg[i] = allavg
#     Spectra[i] = allspect
    
    
# # ''' Plot some examples '''
# # plt.plot(TrigAvg[5].T)    
    
    
# ''' Trial triggered averages of spectra '''
# f, t, spect = scipy.signal.stft(trials, fs=np.squeeze(newfs), nperseg=250, noverlap=240, window='hamming')
# t = t-0.5
# spect = np.mean(abs(spect), axis=0);
# plt.imshow(abs(spect), aspect='auto', extent=[t[0], t[-1], f[-1], f[0]])
# plt.ylim([0, 50])














    
    
    