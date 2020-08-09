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
def getROMData(ptIds, romSize, romWorkDir):
  print('')
  romDirsFullPath = [romWorkDir+'/'+d for d in os.listdir(romWorkDir)
                     if 'rom' in d and str(romSize) in d]
  assert(len(romDirsFullPath)==1)
  # # sort based on the ROM size which is the the last item in dir name
  # def func(elem): return int(elem.split('_')[-1])
  # romDirsFullPath = sorted(romDirsFullPath,key=func)
  # print(romDirsFullPath)

  romDir = romDirsFullPath[0]
  currRomSize = extractRomSizeFromInputFile(romDir)
  assert(romSize==currRomSize)
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
  colormap = plt.cm.get_cmap('BuPu')
  print(colormap(0))
  for i in range(half):
    ax.fill_between(x, P[i,:], P[-(i+1),:],
                    color=colormap(i/half+0.25),
                    #color=color,
                    alpha=alpha,
                    label=str(pct[i])+'th - '+str(pct[-(i+1)])+'th'+' (ROM)',
                    zorder=zorder)

  # # fill lower and upper percentile groups
  # for p1, p2 in zip(perc1, perc2):
  #   ax.fill_between(x, p1, p2, alpha=alpha, color=color,
  #                   edgecolor=None, zorder=zorder, label='2nd/98th percentiles')

  if plot_mean:
      ax.plot(x, np.mean(y, axis=0), color=line_color,
              linewidth=1, zorder=zorder, label='mean (ROM)')
  if plot_median:
      ax.plot(x, np.median(y, axis=0), color=line_color,
              linewidth=1, zorder=zorder)

#=====================================================================
def plotFullSeismogram(idi, data, pct, filename):
  fig, ax = plt.subplots()
  plt.grid('on')
  tsplot(ax, t, data, pct,
         plot_median=False, plot_mean=True, line_color='black',
         alpha=0.5, zorder=5)

  ax.set_xlim([-50, 2050])
  ax.set_ylim([-3e-7, 3e-7])

  ax.set_xticks(np.linspace(0, 2000, 11))
  ax.set_ylabel(r'$v_{\phi}(t)$', fontsize=16)
  ax.set_xlabel(r'Time (seconds)', fontsize=16)
  ax.tick_params(axis='y', which='major', labelsize=12)
  ax.tick_params(axis='y', which='minor', labelsize=12)
  ax.tick_params(axis='x', which='major', labelsize=12)
  ax.tick_params(axis='x', which='minor', labelsize=12)

  if idi == 0:
    box = Rectangle((300,-2.8e-7), 200, 2*2.8e-7,
                    linewidth=1, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)
  if idi == 1:
    box = Rectangle((825,-1.8e-7), 200, 2*1.8e-7,
                    linewidth=1, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)
  if idi == 2:
    box = Rectangle((1600,-1.6e-7), 200, 2*1.6e-7,
                    linewidth=1, edgecolor='k', linestyle='--',
                    facecolor='none')
    ax.add_patch(box)



  plt.legend(loc="upper right", ncol=2, fontsize=10, labelspacing=.3,
             handletextpad=0.05, frameon=False, markerscale=0.75)

  plt.tight_layout()
  fig.savefig(filename+'.png', format="png", bbox_inches='tight', dpi=300)
  fig.savefig(filename+'.pdf', format="pdf", bbox_inches='tight', dpi=300)


#=====================================================================
def plotZoom(idi, data, dataFom, pct, filename):
  [fom_mean, fom_stdev, p1, p2] = extractStatsFromData(dataFom,pct)

  fig, ax = plt.subplots()
  plt.grid('on')
  plt.grid('on')
  tsplot(ax, t, data, pct,
         plot_median=False, plot_mean=True, line_color='black',
         alpha=0.5, zorder=5)

  ax.plot(t, fom_mean, 'o', markersize=2, color='k',
              label='mean (FOM)', markerfacecolor='none', zorder=6)

  ax.errorbar(t, fom_mean, [p1[0,:], p2[-1,:]], capsize=2,
              markeredgewidth=1, elinewidth=0.1, color='k',
              label=str(pct[0])+'th - '+str(pct[-1])+'th'+' (FOM)',
              fmt='none', markerfacecolor='none',
              errorevery=1, zorder=6)

  if int(idi) == 0:
    ax.set_xticks([300, 350, 400, 450, 500, 550])
    ax.set_xlim([300, 500])
    ax.set_ylim([-2.8e-7, 2.8e-7])
  if int(idi) == 1:
    ax.set_xticks([850,900,950,1000,1050])
    ax.set_xlim([825, 1025])
    ax.set_ylim([-1.8e-7, 1.8e-7])
  if int(idi) == 2:
    ax.set_xticks([1600,1650,1700,1750,1800])
    ax.set_xlim([1600, 1800])
    ax.set_ylim([-1.6e-7, 1.6e-7])

  ax.set_ylabel(r'$v_{\phi}(t)$', fontsize=16)
  ax.set_xlabel(r'Time (seconds)', fontsize=16)
  ax.tick_params(axis='y', which='major', labelsize=12)
  ax.tick_params(axis='y', which='minor', labelsize=12)
  ax.tick_params(axis='x', which='major', labelsize=12)
  ax.tick_params(axis='x', which='minor', labelsize=12)
  plt.legend(loc="upper right", ncol=2, fontsize=10, labelspacing=.3,
             handletextpad=0.05, frameon=False, markerscale=0.75)
  plt.tight_layout()

  fig.savefig(filename+'.png', format="png", bbox_inches='tight', dpi=300)
  fig.savefig(filename+'.pdf', format="pdf", bbox_inches='tight', dpi=300)



#=====================================================================
#=====================================================================
romSize = 436
ptId = [0,1,2]
romWorkDir = '.'
fomWorkDir = '.'
#targetPt = ptId[0]

# params for the stats to compute/visualize
percentiles = [5, 10, 25, 75, 90, 95]

[fomK1, t] = getFOMData(ptId, fomWorkDir)
[romK1]    = getROMData(ptId, romSize, romWorkDir)

# loop over point ids
print('loop')
for idi in ptId:
  romData = romK1[idi]
  fomData = fomK1[idi]

  plotFullSeismogram(idi, romData, percentiles, 'full_'+str(idi))
  plotZoom(idi, romData, fomData, percentiles, 'zoom_'+str(idi))
  plt.show()



#fig, ax = plt.subplots()
#ax.errorbar(t, fom_mean, [p1[0,:], p2[-1,:]], capsize=3,
#            markeredgewidth=1, elinewidth=1, fmt='o')
# fig, ax = plt.subplots()
# tsplot(ax, range(numTimeSteps), fomK1[0], n=10, percentile_min=2.5,
#        percentile_max=97.5, plot_median=False, plot_mean=True,
#        color='g', line_color='navy')
# fig1, ax1 = plt.subplots()

#for i in range(fomK1[targetPt].shape[0]):
#  plt.plot(fomK1[targetPt][i,:], '-k', markerfacecolor='none', linewidth=0.5)
#plt.plot(romK1[0][i,:], '--')
