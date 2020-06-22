#!/usr/bin/env python

import sys, os, time
import subprocess, math
import numpy as np
import os.path, re
from scipy import stats
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, linewidth=100000)

# for mrsteam
peakMemBW = 84 #GB

nf          = [2, 4, 8, 18, 36]
nfStr       = ['2', '4', '8', '18', '36']

nThreads    = [2, 4, 8, 18, 36, 72]
nThreadsStr = ['2', '4', '8', '18', '36', '72']
hatches     = {'2': 'ooooo',
               '4': '\\\\',
               '8': '/////',
               '18': '||||',
               '36': 'xxxxx',
               '72': '---'}

# num of dofs for each case for plotting
#785152, 3143168, 12577792, 50321408
meshLabelsPlot = [r'0.8M', r'3M', r'12M', r'50M']

#=====================================================================
def extractThreadsList(data):
  return np.unique(data[:, 0])

#=====================================================================
def createDicByNumThreads(data, metric, stat):
  # create a dictionary where
  # key = num threads
  # value = array with values for each numOfthread (in increasing order)

  assert(metric=="mem" or metric=="cpu")
  if metric == "mem":
    if stat == "ave": targetCol = 4
    elif stat=="min": targetCol = 5
    elif stat=="max": targetCol = 6

  elif metric == "cpu":
    if stat == "ave": targetCol = 7
    elif stat=="min": targetCol = 8
    elif stat=="max": targetCol = 9

  dic = {}
  for i in range(data.shape[0]):
    # get size
    thisSize = str(int(data[i][0]))
    # get target metric value
    value = data[i][targetCol]

    # we loop over rows of the data file, and the rows of the data file
    # are in increasing order from threads=2,4,..., so by appending we keep the order
    if thisSize in dic: dic[thisSize].append(value)
    else: dic[thisSize] = [value]
  return dic

def plotBars(d, ax, xref):
  nTh, nf = d[:,0], d[:,1]
  mm = d[:,4]
  n = len(nTh)
  x = np.ones(n)*xref
  y = np.arange(len(nf))
  z = np.zeros(n)
  dx = np.ones(n)*0.8
  dy = np.ones(len(nf))*0.25
  dz = mm
  ax.bar3d(x, y, z, dx, dy, dz, alpha=1, linewidth=2)

#=====================================================================
def main(dataFile, metric, stat):
  data = np.loadtxt(dataFile)
  #dataDic = createDicByNumThreads(data, metric, stat)
  #plotBar(dataDic, meshLabelsPlot, nThreads, nThreadsStr, metric, stat)

  fig = plt.figure()
  ax1 = fig.add_subplot(111, projection='3d')

  numThreadsCases = len(nThreads)
  numFCases = len(nf)
  dic = {0: 24,
         30: 16,
         60: 8,
         90: 0}
  for shift in [0, 30, 60, 90]:
    rS = shift
    for i in range(numThreadsCases):
      plotBars(data[rS:rS+numFCases, :], ax1, i+dic[shift])
      print(shift, i, rS)
      rS += numThreadsCases-1

  #ax1.set_zscale('log')
  ax1.view_init(elev=30., azim=-131)
  plt.show()


#////////////////////////////////////////////
if __name__== "__main__":
#////////////////////////////////////////////
  parser = ArgumentParser()
  parser.add_argument("-file", "--file",
                      dest="dataFile",
                      help="where to get data from\n")

  parser.add_argument("-metric", "--metric",
                      dest="metric", default="memBW",
                      help="mem or cpu\n")

  parser.add_argument("-stat", "--stat",
                      dest="stat", default="ave",
                      help="ave, min or max\n")

  args = parser.parse_args()
  main(args.dataFile, args.metric, args.stat)
#////////////////////////////////////////////
