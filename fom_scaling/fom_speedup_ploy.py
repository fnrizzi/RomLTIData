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

targetThreads = [2, 4, 12, 18, 36]
fomIDToDofs   = {0: 785152, 1: 3143168, 2: 12577792, 3: 50321408}

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
  return dataRow[13]

#=====================================================================
def findNormalizingValue(data, numDofs, nThr, N):
  # we want to assess when computing multiple forcing at same time
  # becomes more convenient than using f=1
  # To to this, we want to find a normalization value to use
  # for all cases to highlight speedup.

  # The normalizing value is = total time I would need to
  # complete N forcing samples using ONLY individual 2-threaded runs with f=1.
  # This means that if I have nThr threads available, the best I can do with the
  # constraint of using f=1 is to do nThr/2 runs going in parallel, each of which uses two threads.
  # in this sceario, how many sets of runs do I need to do to obtain a total of N samples?
  #    one set comprises nThr/2 runs, each with f=1, so each set yields nThr/2 forcing samples.
  #   so to do N samples, I need: N/(nThr/2)

  setsOfRuns = float(2.*N/nThr)

  for i in range(data.shape[0]):
    thisNumThr   = int(data[i][0])
    thisValF     = int(data[i][1])
    thisNumDofs  = int(data[i][2])

    # find the case where f=1 and threads=2
    if thisNumDofs==numDofs and thisValF==1 and thisNumThr==2:
      loopTime = extractLoopTime(data[i,:])

      # since we have nThr/2 runs going in parallel each with f=1,
      # the time to complete the runs in one sets = time to complete one run
      # so the time to complete all sets is:
      return loopTime * setsOfRuns
  sys.exit("Did not find a normalizing value")

#=====================================================================
def computeData(dataIn, numDofs, nThr, N):
  # dataOut is a a matrix where:
  # - rows    indexing threads
  # - columns indexing values of f

  thCases = targetThreads
  print(thCases)
  # extract from data all unique values of f
  fCases  = extractAllFCases(dataIn)

  # create data matrix
  dataOut = np.zeros((len(thCases), len(fCases)))

  # find normalizing value for the speedup
  normalizValue = findNormalizingValue(dataIn, numDofs, nThr, N)
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
    thisNumDofs  = int(dataIn[i][2])

    # get the loop time for this case
    thisLoopTime = extractLoopTime(dataIn[i,:])

    if (thisNumDofs==numDofs and thisNumThr in thCases):
      # how many concurrent runs can I do?
      numConcurrentRuns = float(nThr/thisNumThr)
      #
      myValue      = float( N/(numConcurrentRuns*thisValF) ) * thisLoopTime

      # find where this entry goes in the matrix
      i,j = thToRowInd[thisNumThr], fToColInd[thisValF]

      # set constraint (e.g. memory) such that I cannot have more
      # than 48 concurrent forcings being evaluated, so if that is the case,
      # set to zero to indicate this is not a feasible case
      if numConcurrentRuns*thisValF > 48:
        dataOut[i][j] = 0.
      else:
        # compute speedup
        dataOut[i][j] = normalizValue/myValue

  return [dataOut, thCases, fCases]


#=====================================================================
def do2dPlot(dataMatrix, thCases, fCases, numDofs, nThr, N):
  #dataMatrix = np.flipud(dataMatrix)

  fig = plt.figure()
  mask = np.zeros_like(dataMatrix)
  mask[dataMatrix==0] = True
  cm = plt.cm.get_cmap('PuBuGn')#.reversed()

  ax = sea.heatmap(dataMatrix, annot=True, center=8,
                   annot_kws={"size": 13},
                   fmt="3.2f", linecolor='white', vmin=0, vmax=27,
                   linewidths=.25, mask=mask, cmap=cm,
                   cbar_kws={'label': 's(f,n)'})

  ax.figure.axes[-1].yaxis.label.set_size(15)

  nR = dataMatrix.shape[0]
  nC = dataMatrix.shape[1]

  ax.set_xticks(np.arange(1, nC+1, 1)-0.5)
  xlab = [str(int(p)) for p in fCases]
  ax.set_xticklabels(xlab, fontsize=14)
  ax.set_xlabel('Size of f', fontsize=16)

  ax.set_yticks(np.arange(1, nR+1, 1)-0.5)
  #ylab = [str(int(p)) for p in thCases[::-1]]
  ylab = [str(int(p)) for p in thCases]
  ax.set_yticklabels(ylab, fontsize=14)
  ax.set_ylabel('Number of threads', fontsize=16)
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

  plt.tight_layout()
  fileName = 'fom_speedup_numDofs_'+str(numDofs)+'_nth_'+str(nThr)+'_N_'+str(N)+'.png'
  fig.savefig(fileName, format="png",
              bbox_inches='tight', dpi=300)

#=====================================================================
def main(dataFile, fomID, nThr, N):
  data = np.loadtxt(dataFile)
  numDofs = fomIDToDofs[fomID]
  [M, thCases, fCases] = computeData(data, numDofs, nThr, N)
  print(M)

  do2dPlot(M, thCases, fCases, numDofs, nThr, N)


#////////////////////////////////////////////
if __name__== "__main__":
#////////////////////////////////////////////
  parser = ArgumentParser()
  # parser.add_argument("-file", "--file",
  #                     dest="dataFile",
  #                     help="where to get data from\n")

  parser.add_argument("-fom-id", "--fom-id", "-fomid", "--fomid",
                      dest="fomID", default=1, type=int,
                      help="Fom case id: 0,1,2,3\n")

  parser.add_argument("-thread-budget", "--thread-budget", "-thrB", "--thrB",
                      dest="nThr", default=36, type=int,
                      help="Budget of threads\n")

  parser.add_argument("-target-samples", "--target-samples", "-targetS", "--targetS",
                      dest="numSamp", default=8192, type=int,
                      help="Target samples to execute\n")

  args = parser.parse_args()

  # assert that the num of threads available is == max value inside
  # the targetThreads list at the top since it would not make sense
  assert( int(args.nThr) == np.max(targetThreads) )

  # we cannot have negative args
  assert( args.numSamp>0 and args.nThr>0 and args.fomID in [0,1,2,3])
  dataFile = './fom_scaling_final.txt'
  main(dataFile, int(args.fomID), int(args.nThr), int(args.numSamp))

#////////////////////////////////////////////
