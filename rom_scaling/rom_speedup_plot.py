#!/usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
import sys, os, time, yaml
import pprint as pp
import subprocess, math
import numpy as np
import os.path, re
from scipy import stats
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sea
from matplotlib.patches import Rectangle

np.set_printoptions(edgeitems=10, linewidth=100000)

targetThreads = [1, 2, 4, 12, 18, 36]

#=====================================================================
def extractAllFCases(data):
  return np.unique(data[:,1])

#=====================================================================
def createRowAndColMappers(thCases, fCases):
  fToColInd, j = {}, 0
  for f in fCases:
    fToColInd[int(f)] = j
    j+=1

  thToRowInd, i = {}, 0
  for th in thCases:
    thToRowInd[int(th)] = i
    i+=1

  return [thToRowInd, fToColInd]

#=====================================================================
def extractLoopTime(dataRow):
  # the loop time is stored in col 14
  return dataRow[14]

#=====================================================================
def findNormalizingValue(data, romSize, nThr, N):
  # we want to assess whether using multiple forcing at same time
  # and using multiple threads is more convenient than using f=1

  # To to this, we want to find a normalization value to use
  # for all cases to highlight speedup.

  # The normalizing value is = total time I would need to
  # complete N forcing samples using ONLY runs with f=1 single-threaded runs.
  # This means that if I have nThr threads available, the best I can do
  # with the constraing of using f=1 is to schedule/run nThr parallel runs
  # each of which uses a single thread.
  # in this sceario, how many sets of runs do I need to do?
  #   I need: N/nThr  because we do nThr runs in parallel at same time
  # so each set allows me to complete nThr samples.
  setsOfRuns = float(N/(nThr/1))

  for i in range(data.shape[0]):
    thisNumThr   = int(data[i][0])
    thisValF     = int(data[i][1])
    thisNumModes = int(data[i][2])

    # find the case where f=1 and threads=1
    if thisNumModes==romSize and thisValF==1 and thisNumThr==1:
      loopTime = extractLoopTime(data[i,:])

      # since we have running in parallel nThr runs each with f=1, the time of doing
      # to complete one set = time of doing one single run
      return loopTime * setsOfRuns
  sys.exit("Did not find a normalizing value")


#=====================================================================
def computeData(dataIn, romSize, nThr, N):
  # dataOut is a a matrix where:
  # - rows    indexing threads
  # - columns indexing values of f

  # extract from data all unique values of threads and f
  thCases = targetThreads #extractAllThreadCases(dataIn)
  fCases  = extractAllFCases(dataIn)
  print(thCases)

  # create matrix
  dataOut = np.zeros((len(thCases), len(fCases)))

  # find normalizing value for the speedup
  normalizValue = findNormalizingValue(dataIn, romSize, nThr, N)
  print(normalizValue)

  # create dic to map values {thread,f} to their {row,col} indeces
  # this is needed to fill the data below
  [thToRowInd, fToColInd] = createRowAndColMappers(thCases, fCases)
  print(thToRowInd)
  print(fToColInd)

  for i in range(dataIn.shape[0]):
    # number of threads and number of modes
    thisNumThr   = int(dataIn[i][0])
    thisValF     = int(dataIn[i][1])
    thisNumModes = int(dataIn[i][2])

    # get the loop time for this case
    thisLoopTime = extractLoopTime(dataIn[i,:])

    if (thisNumModes==romSize and thisNumThr in thCases):
      # how many concurrent runs can I do?
      numConcurrentRuns = float(nThr/thisNumThr)
      #
      myValue      = float( N/(numConcurrentRuns*thisValF) ) * thisLoopTime

      # find where this entry goes in the matrix
      i,j = thToRowInd[thisNumThr], fToColInd[thisValF]

      # set constraint (e.g. memory) such that I cannot have more
      # than 1024 concurrent forcings being evaluated, so if that is the case,
      # set to zero to indicate this is not a feasible case
      if numConcurrentRuns*thisValF > 1024:
        dataOut[i][j] = 0.
      else:
        # compute speedup
        dataOut[i][j] = normalizValue/myValue

  return [dataOut, thCases, fCases]


#=====================================================================
def do2dPlot(dataMatrix, thCases, fCases, romSize, nThr, N):
  #dataMatrix = np.flipud(dataMatrix)

  fig = plt.figure()
  mask = np.zeros_like(dataMatrix)
  mask[dataMatrix==0] = True
  cm = plt.cm.get_cmap('PuBuGn')#.reversed()

  thislabel = r'$s($'+str(romSize)+r'$, n, M)$'
  ax = sea.heatmap(dataMatrix, annot=True, center=8, annot_kws={"size": 9},
                   fmt="3.2f", linecolor='white', vmin=0, vmax=27,
                   linewidths=.25, mask=mask, cmap=cm,
                   cbar_kws={'label': thislabel})

  ax.figure.axes[-1].yaxis.label.set_size(15)

  nR = dataMatrix.shape[0]
  nC = dataMatrix.shape[1]

  ax.set_xticks(np.arange(1, nC+1, 1)-0.5)
  xlab = [str(int(p)) for p in fCases]
  ax.set_xticklabels(xlab, fontsize=11)
  ax.set_xlabel(r'$M$', fontsize=16)

  ax.set_yticks(np.arange(1, nR+1, 1)-0.5)
  #ylab = [str(int(p)) for p in thCases[::-1]]
  ylab = [str(int(p)) for p in thCases]
  ax.set_yticklabels(ylab, fontsize=12)
  ax.set_ylabel(r'$n$ (Number of threads)', fontsize=16)
  #plt.rcParams['axes.linewidth'] = 1

  for i in range(dataMatrix.shape[0]):
    for j in range(dataMatrix.shape[1]):
      if (j>=1):
        if dataMatrix[i][j-1] <= 1. and dataMatrix[i][j]>1.:
          ax.add_patch(Rectangle((j,i), 0., 1,
                                 edgecolor='black', fill=False, lw=1.5, zorder=10))
      if (i<nR-1):
        if dataMatrix[i+1][j] <= 1. and dataMatrix[i][j]>1.:
          ax.add_patch(Rectangle((j,i+1), 1, 0,
                                 edgecolor='black', fill=False, lw=1.5, zorder=10))

  #ax.add_patch(Rectangle((0.025,5),0.975,0.98,edgecolor='black', fill=False, lw=1.5, zorder=10))

  plt.tight_layout()
  fileName = 'rom_speedup_romSize_'+str(romSize)+'_nth_'+str(nThr)+'_N_'+str(N)+'.png'
  fig.savefig('./plots/'+fileName, format="png",
              bbox_inches='tight', dpi=300)

#=====================================================================
def do3dPlot(dataMatrix, thCases, fCases):
  fig = plt.figure(0)
  ax = Axes3D(fig)
  lx= len(fCases)
  ly= len(thCases)
  xpos = np.arange(0,lx,1)    # Set up a mesh of positions
  ypos = np.arange(0,ly,1)
  xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
  xpos = xpos.flatten()   # Convert positions to 1D array
  ypos = ypos.flatten()
  zpos = np.zeros(lx*ly)
  dx = 0.5 * np.ones_like(zpos)
  dy = dx.copy()
  dz = dataMatrix.flatten()
  ax.bar3d(xpos,ypos,zpos, dx, dy, dz)


#=====================================================================
def main(dataFile, romSize, nThr, N):
  data = np.loadtxt(dataFile)
  [M, thCases, fCases] = computeData(data, romSize, nThr, N)
  print(M)

  do2dPlot(M, thCases, fCases, romSize, nThr, N)
  #do3dPlot(M, thCases, fCases)
  #plt.show()


#////////////////////////////////////////////
if __name__== "__main__":
#////////////////////////////////////////////
  parser = ArgumentParser()
  # parser.add_argument("-file", "--file",
  #                     dest="dataFile",
  #                     help="where to get data from\n")

  parser.add_argument("-rom-size", "--rom-size", "-romsize", "--romsize",
                      dest="romSize", default=1024, type=int,
                      help="Rom size to use, choices: 256, 512, 1024, 2048, 4096\n")

  parser.add_argument("-thread-budget", "--thread-budget", "-thrB", "--thrB",
                      dest="nThr", default=36, type=int,
                      help="Budget of threads\n")

  parser.add_argument("-target-samples", "--target-samples", "-targetS", "--targetS",
                      dest="numSamp", default=8192, type=int,
                      help="Target samples to execute\n")

  args = parser.parse_args()

  # assert that the num of threads available is not smaller than
  # max value inside the targetThreads list at the top since it would not make sense
  assert( int(args.nThr) == np.max(targetThreads) )

  # we cannot have negative args
  assert( args.numSamp>0 and args.nThr>0 and args.romSize>0)
  dataFile = './data/rom_scaling_final.txt'
  main(dataFile, int(args.romSize), int(args.nThr), int(args.numSamp))
#////////////////////////////////////////////
