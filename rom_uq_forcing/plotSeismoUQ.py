#!/usr/bin/env python

import struct
import numpy as np
import sys, re, os, yaml, glob
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy import linalg as la
from matplotlib import cm
from matplotlib.patches import Rectangle

#=========================================
def extractNumStepsStoredSeismo(dirPath):
  with open(dirPath+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    dt = float(ifile['general']['dt'])
    totsteps = float(ifile['general']['finalTime'])
    seismoFreq = int(ifile['io']['seismogram']['freq'])
    return int(totsteps/dt)/seismoFreq

#=========================================
def extractFreqStoredSeismo(dirPath):
  with open(dirPath+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    seismoFreq = int(ifile['io']['seismogram']['freq'])
    return seismoFreq

#=========================================
def extractDt(dirPath):
  with open(dirPath+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    return float(ifile['general']['dt'])

#=========================================
def extractRomSizeFromInputFile(dirPath):
  with open(dirPath+'/input.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    return int( ifile['rom']['velocity']['numModes'] )

#=========================================
def extractForcingSizeFromInputFile(dirPath):
  with open(dirPath+'/input.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    return int( ifile['sampling']['forcingSize'] )

#=========================================
def extractNumOfReceivers(dirPath):
  with open(dirPath+'/input.yaml') as file:
    ifile = yaml.load(file, Loader=yaml.FullLoader)
    recList = ifile['io']['seismogram']['receivers']
    return int(len(recList))

#=====================================================================
def getDataSingleID(ptId, dataDir, forcingSize, numSets):
  numSamples = forcingSize*numSets
  D = np.zeros((numSamples,1))
  k=0
  for i in np.arange(forcingSize):
    for j in np.arange(numSets):
      fileToLoad = dataDir+'/seismo_f_'+str(i)+'_'+str(j)
      tmpD = np.loadtxt(fileToLoad, skiprows=1)
      nCols = tmpD.shape[1]
      if D.shape[1] == 1: D.resize((numSamples, nCols))
      D[k,:] = tmpD[ptId,:]
      k+=1
  return D

#=====================================================================
def getROMData(ptIds, romSizeVp, romWorkDir):
  print('')
  romDirsFullPath = [romWorkDir+'/'+d for d in os.listdir(romWorkDir)
                     if 'rom' in d and str(romSizeVp) in d
                     and 'nThreads_4' in d]
  assert(len(romDirsFullPath)==1)

  romDir = romDirsFullPath[0]
  currRomSize = extractRomSizeFromInputFile(romDir)
  assert(romSizeVp==currRomSize)
  print('ROM: size={}'.format(currRomSize))
  # extract the forcing size
  currForcingSize = extractForcingSizeFromInputFile(romDir)
  print('ROM: forcingSize={}'.format(currForcingSize))

  # sets of runs (find from number of seismo_ files)
  numSets0 = len(glob.glob(romDir+"/seismo_f_0_*"))
  numSets  = 1 if numSets0==0 else numSets0
  print('ROM: numSets={}'.format(numSets))

  # dataDic:, key=ptId, value=numpyarray with all realizations
  dataDic = {}
  for pt in ptIds:
    dataDic[pt] = getDataSingleID(pt, romDir, currForcingSize, numSets)

  #firstKey = next(iter(dataDic))
  return [dataDic] #, dataDic[firstKey].shape[1]]


#=====================================================================
def getFOMData(ptIds, fomWorkDir):
  fomDirsFullPath = [fomWorkDir+'/'+d for d in os.listdir(romWorkDir)
                     if 'fom' in d and 'test' in d]
  assert(len(fomDirsFullPath)==1)
  fomDir = fomDirsFullPath[0]

  # extract the forcing size
  currForcingSize = extractForcingSizeFromInputFile(fomDir)
  print('FOM: forcingSize={}'.format(currForcingSize))
  # sets of runs (find from number of seismogram_ files)
  seismoFiles = glob.glob(fomDir+"/seismogram_*")
  numSets = len(seismoFiles)
  print('FOM: numSets={}'.format(numSets))
  # num of receivers
  numRec  = extractNumOfReceivers(fomDir)
  print('FOM: numReceivers={}'.format(numRec))
  # totaly num samples
  numSamples = currForcingSize*numSets

  numCols = int(extractNumStepsStoredSeismo(fomDir))
  print('FOM: numSeismoTimeSteps={}'.format(numCols))

  # dataDic: key=ptId, value=numpyarray with all realizations
  dataDic = {}
  for pt in ptIds:
    dataDic[pt] = np.zeros((numSamples,numCols))

  for ise in range(numSets):
    fileToLoad = fomDir+'/seismogram_'+str(ise)
    tmpD = np.loadtxt(fileToLoad, skiprows=0)
    # for each file loaded, need to loop over all
    # realizations for every pt
    startWriteRow = ise*currForcingSize
    for pt in ptIds:
      index,k = pt, 0
      for j in range(currForcingSize):
        newReading = tmpD[index,:]
        dataDic[pt][startWriteRow+k, :] = newReading
        index += numRec
        k+=1

  # the times where we collected the seismogram
  freq = extractFreqStoredSeismo(fomDir)
  dt = extractDt(fomDir)
  t = freq*dt*np.arange(numCols)
  return [dataDic, t]

#=====================================================================
def extractStatsFromData(D, pct): #percentile_min=1, percentile_max=99, n=20):
  s1 = np.mean(D, axis=0)
  s2 = np.std(D, axis=0)

  half = int((len(pct)-1)/2) if len(pct) % 2 != 0 else int(len(pct)/2)
  p1 = np.percentile(D, pct[:half], axis=0)
  p2 = np.percentile(D, pct[half:], axis=0)
  #np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
  #p2 = np.percentile(D, pct, axis=0)
  #np.linspace(50, percentile_max, num=n+1)[1:], axis=0)

  return [s1, s2, s1-p1, p2-s1]

#=====================================================================
def tsplot(ax, x, y, pct, color='r',
           plot_mean=True, plot_median=False, line_color='k',
           alpha = 0.5, **kwargs):

  # # calculate the lower and upper percentile groups, skipping 50 percentile
  # p1vals = np.linspace(percentile_min, 50, num=n, endpoint=False)
  # p2vals = np.linspace(50, percentile_max, num=n+1)[1:]
  # print(p1vals)
  # print(p2vals)

  if 'zorder' in kwargs: zorder = kwargs.pop('zorder')
  #if 'alpha' in kwargs: alpha = kwargs.pop('alpha')
  #else: alpha = 1/n

  P = np.percentile(y, pct[:], axis=0)

  half = int((len(pct)-1)/2) if len(pct) % 2 != 0 else int(len(pct)/2)
  #colormap = plt.cm.get_cmap('BuPu')
  #print(colormap(0))
  #mycolors = ['#264653', '#2a9d8f', ]
  mycolors = ['#d62828', '#f77f00', '#e9c46a']

  for i in range(half):
    ax.fill_between(x, P[i,:], P[-(i+1),:],
                    color=mycolors[i], #colormap(i/half+0.65),
                    #color=color,
                    alpha=alpha,
                    label=str(pct[i])+'th - '+str(pct[-(i+1)])+'th',
                    zorder=zorder)

  # # fill lower and upper percentile groups
  # for p1, p2 in zip(perc1, perc2):
  #   ax.fill_between(x, p1, p2, alpha=alpha, color=color,
  #                   edgecolor=None, zorder=zorder, label='2nd/98th percentiles')

  if plot_mean:
      ax.plot(x, np.mean(y, axis=0), color=line_color,
              linewidth=1, zorder=zorder, label='mean')
  if plot_median:
      ax.plot(x, np.median(y, axis=0), color=line_color,
              linewidth=1, zorder=zorder)

#=====================================================================
def plotFullSeismogram(idi, data, pct, filename):
  fig, ax = plt.subplots()
  plt.grid('on')
  tsplot(ax, t, data, [pct[0], pct[-1]],
         plot_median=False, plot_mean=True, line_color='black',
         alpha=0.7, zorder=5)

  ax.set_xlim([-50, 2050])
  ax.set_ylim([-3e-7, 3e-7])

  ax.set_yticks(np.linspace(-3e-7, 3e-7, 13))
  ax.set_xticks(np.linspace(0, 2000, 6))
  # if idi==0: ylab = r'$v_{\phi}(r_{earth}, 0.174533, t)$'
  # if idi==1: ylab = r'$v_{\phi}(r_{earth}, \pi/6, t)$'
  # if idi==2: ylab = r'$v_{\phi}(r_{earth}, 2\pi/6, t)$'
  ylab = r'$v_{\phi}(t)$'
  ax.set_ylabel(ylab, fontsize=18)
  ax.set_xlabel(r'Time (seconds)', fontsize=18)
  ax.tick_params(axis='y', which='major', labelsize=15)
  ax.tick_params(axis='y', which='minor', labelsize=15)
  ax.tick_params(axis='x', which='major', labelsize=15)
  ax.tick_params(axis='x', which='minor', labelsize=15)

  if idi == 0:
    box = Rectangle((300,-2.7e-7), 200, 2*2.7e-7,
                    linewidth=1.5, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)
  if idi == 1:
    box = Rectangle((800,-2e-7), 200, 2*2e-7,
                    linewidth=1.5, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)
  if idi == 2:
    box = Rectangle((1600,-1.8e-7), 200, 2*1.8e-7,
                    linewidth=1.5, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)

  if idi==0:
    plt.legend(loc="lower right", ncol=1, fontsize=15, labelspacing=.3,
               handletextpad=0.2, frameon=False, markerscale=0.75)

  plt.tight_layout()
  fig.savefig('./plots/'+filename+'.png', format="png", bbox_inches='tight', dpi=300)


#=====================================================================
def plotZoom(idi, data, dataFom, pct, filename):
  [fom_mean, fom_stdev, p1, p2] = extractStatsFromData(dataFom,pct)

  fig, ax = plt.subplots()
  plt.grid('on')
  plt.grid('on')
  tsplot(ax, t, data, pct,
         plot_median=False, plot_mean=True, line_color='black',
         alpha=0.7, zorder=5)

  ax.plot(t, fom_mean, 'o', markersize=2, color='k',
              label='mean (FOM)', markerfacecolor='none', zorder=6)

  ax.errorbar(t, fom_mean, [p1[0,:], p2[-1,:]], capsize=2,
              markeredgewidth=1, elinewidth=0.1, color='k',
              label=str(pct[0])+'th - '+str(pct[-1])+'th'+' (FOM)',
              fmt='none', markerfacecolor='none',
              errorevery=1, zorder=6)

  ax.set_yticks(np.linspace(-3e-7, 3e-7, 13))
  if int(idi) == 0:
    ax.set_xticks([300, 350, 400, 450, 500, 550])
    ax.set_xlim([300, 500])
    ax.set_ylim([-2.7e-7, 2.7e-7])
  if int(idi) == 1:
    ax.set_xticks([800,850,900,950,1000])
    ax.set_xlim([800, 1000])
    ax.set_ylim([-2e-7, 2e-7])
  if int(idi) == 2:
    ax.set_xticks([1600,1650,1700,1750,1800])
    ax.set_xlim([1600, 1800])
    ax.set_ylim([-1.8e-7, 2e-7])

  ax.set_ylabel(r'$v_{\phi}(t)$', fontsize=18)
  ax.set_xlabel(r'Time (seconds)', fontsize=18)
  ax.tick_params(axis='y', which='major', labelsize=15)
  ax.tick_params(axis='y', which='minor', labelsize=15)
  ax.tick_params(axis='x', which='major', labelsize=15)
  ax.tick_params(axis='x', which='minor', labelsize=15)
  if idi==0:
    plt.legend(loc="lower right", ncol=1, fontsize=14, labelspacing=.2,
               borderpad=0, handletextpad=0.2, frameon=False, markerscale=0.75)

  plt.tight_layout()
  fig.savefig('./plots/'+filename+'.png', format="png", bbox_inches='tight', dpi=300)


#=====================================================================
#=====================================================================
romSizeVp = 436
ptId = [0,1,2]
romWorkDir = '.'
fomWorkDir = '.'

percentiles = [5, 10, 25, 75, 90, 95]

[fomK1, t] = getFOMData(ptId, fomWorkDir)
[romK1]    = getROMData(ptId, romSizeVp, romWorkDir)

# loop over point ids
print('loop')
for idi in ptId:
  romData = romK1[idi]
  fomData = fomK1[idi]

  plotFullSeismogram(idi, romData, percentiles, 'full_'+str(idi))
  plotZoom(idi, romData, fomData, percentiles, 'zoom_'+str(idi))
  plt.show()
