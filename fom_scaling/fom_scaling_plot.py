#!/usr/bin/env python

import sys, os, time, yaml
import pprint as pp
import subprocess, math
import numpy as np
import os.path, re
from scipy import stats
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, linewidth=100000)

# peak BW for mrstem (GB) = 21333.33 * 4 at time of doing these runs
peakMemBW = 85.

myAlpha = .9

# num of dofs for each case for plotting
#785152, 3143168, 12577792, 50321408
meshLabelsPlot = [r'0.78M', r'3M', r'12M', r'50M']

nThreads    = [2, 8, 36]

fSizes      = [1,4,8,32,48]
colors   = {1:  '#003f5c',
            4:  '#ffa600',
            8:  '#ef5675',
            32: '#7a5195',
            48: '#00818a'}

#=====================================================================
def computeMetricValue(lineData, currValueF, metric, stat):
  if metric == "mem":
    if stat == "ave": return lineData[4]
    elif stat=="min": return lineData[5]
    elif stat=="max": return lineData[6]

  elif metric == "cpu":
    if stat == "ave": return lineData[7]
    elif stat=="min": return lineData[8]
    elif stat=="max": return lineData[9]

  elif metric == "itertime":
    if stat == "ave": return lineData[10]
    elif stat=="min": return lineData[11]
    elif stat=="max": return lineData[12]

  elif metric == "looptime":
    return lineData[13]

#=====================================================================
def createDataDic(data, metric, stat):
  # create a dictionary where
  # key = num threads
  # value = array with values for each numOfthread (in increasing order)

  all = {}
  # for every value of f, create a dic based on the # of threads
  for f in fSizes:
    dic = {}
    for i in range(data.shape[0]):
      # number of threads and number of modes
      thisNumThr   = int(data[i][0])
      thisValF     = int(data[i][1])

      if thisValF == f:
        if thisNumThr in nThreads:
          value = computeMetricValue(data[i,:], thisValF, metric, stat)

          if thisNumThr in dic: dic[thisNumThr].append(value)
          else: dic[thisNumThr] = [value]
    all[f] = dic

  return all

#=====================================================================
def plotBarSet(ax, xLoc, width, nThr, dic, myColor):
  val = dic[nThr]
  ax.bar(xLoc, val, width, alpha=myAlpha, color=myColor, edgecolor='none', zorder=5)

#=====================================================================
def plotBar(dataDic, meshLabels, nThreads, metric, stat):
  # number of mesh sizes to deal with
  numMeshes = len(meshLabels)
  # Setting the positions and width for the bars
  posArray = range(numMeshes)
  pos = list(posArray)

  width = 0.35
  plt.rc('axes', axisbelow=True)

  fig, ax = plt.subplots(figsize=(9,6))
  plt.grid()
  ax2 = ax.twiny()
  fig.subplots_adjust(bottom=0.25)

  gigi = [0.2, 6.25, 12.3, 18.3]

  #---- loop over numthreads and plot ----
  xTicksBars, xTlabels = [], []
  count=0
  for k,v in dataDic.items():
    for i,nThr in enumerate(nThreads):
      #x locations for the bars
      shift = width*i*5.8

      xLoc = [p+shift+0.36*count+gigi[k] for k,p in enumerate(pos)]

      plotBarSet(ax, xLoc, width, nThr, v, colors[k])
      xTicksBars += [p+shift+0.75+gigi[k] for k,p in enumerate(pos)]
      xTlabels += [str(nThr) for i in range(numMeshes)]
    count+=1

  # hack to add labels for legend, need as many as values of f
  for iF in fSizes:
    ax.bar(100, 1, width, alpha=myAlpha, color=colors[iF],
           edgecolor='none', zorder=-1, label='f='+str(iF))

  ax.legend(loc="upper left", ncol=5, fontsize=13, frameon=False)

  # remove the vertical lines of the grid
  ax.xaxis.grid(which="major", color='None', linestyle='-.', linewidth=0, zorder=0)

  ax.xaxis.set_ticks_position('bottom')
  ax.xaxis.set_label_position('bottom')
  ax.set_xticks(xTicksBars)
  ax.set_xticklabels(xTlabels, fontsize=15)
  ax.xaxis.set_tick_params(rotation=0)

  ax.set_xlabel(r'Number of threads', fontsize=16)
  ax.set_xlim(min(pos)-0.2, max(pos)+width*69.3)

  if metric =="mem":
    ax.set_yscale('log')
    ax.set_ylabel("Memory Bandwith (GB/s)", fontsize=18)
    ax.set_ylim([1e-1, 1000])
    ax.set_yticks([1e-1, 1, 10, 100, 1000])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)

    # plot peak theoretical mem BW
    ax.plot([min(pos)-0.2, max(pos)+width*70],
            [peakMemBW, peakMemBW], '--k', linewidth=1.2, zorder=7)
    ax.text((min(pos)+width+max(pos)+width*75)*0.5,
            peakMemBW+12, 'Machine\'s theoretical peak', fontsize=15)

  elif metric=='cpu':
    ax.set_yscale('log')
    ax.set_ylabel("GFlops", fontsize=18)
    ax.set_ylim([1e-1, 1e4])
    ax.set_yticks([1e-1, 1, 10, 1e2, 1e3, 1e4])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)

  elif metric =="itertime":
    ax.set_yscale('log')
    ax.set_ylim([1e-1, 1e4])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)
    if stat == 'ave': pref = 'Average'
    elif stat=='min': pref = 'Min'
    elif stat=='max': pref = 'Max'
    ax.set_ylabel(pref+" time (ms)/timestep", fontsize=18)


  # ticks for the meshes
  meshTicks = [3.1, 9.85, 16.75, 23.5]
  ax2.set_xticks(meshTicks)
  ax2.xaxis.set_ticks_position('bottom')
  ax2.xaxis.set_label_position('bottom')
  ax2.spines['bottom'].set_position(('outward', 65))
  ax2.set_xlabel('Problem Size (degrees of freedom)', fontsize=16)
  ax2.set_xticklabels(meshLabels, fontsize=16)
  ax2.set_xlim(min(pos), max(pos)+width*68)
  ax2.set_axisbelow(True)

  plt.tight_layout()
  fileName = "fom_"+metric+"_"+stat+".png"
  fig.savefig(fileName, format="png",bbox_inches='tight', dpi=300)
  plt.show()


#=====================================================================
def plotLines(dataDic, meshLabels, nThreads, metric, stat):
  meshSet = [0, 3]
  numMeshes = len(meshSet)

  mk = ['P', 's', 'o']
  lt = ['-', '-',  '-']

  fig, ax = plt.subplots()
  plt.grid()
  for k,v in dataDic.items():
    # k is the value of f
    # v is a dic with key=nThreads, values=array with times for all meshes
    thisF = k

    if (thisF in [1,8,48]):
      # create vector with times for all meshes
      for iM, M in enumerate(meshSet):
        thisData = np.array([v1[M] for k1,v1 in v.items()])
        plt.plot(nThreads, thisData, lt[iM], marker=mk[iM], markerfacecolor='none',
                 color=colors[thisF], linewidth=1.5, markersize=8)

  # hack for legend
  for iM, M in enumerate(meshSet):
    plt.plot(100, 100, '-', marker=mk[iM], color='k', linewidth=0, markersize=7, label=meshLabelsPlot[M])
  for iF in [1,8,48]:
    plt.plot(100, 100, '-', marker='*', color=colors[iF], linewidth=1.2, markersize=1, label='f='+str(iF))


  ax.legend(loc="upper right", ncol=3, fontsize=13, frameon=False)
  ax.set_xticks(nThreads)
  ax.set_yscale('log')

  ax.set_xlabel(r'Number of threads', fontsize=16)
  ax.set_xlim(-5, 80)
  ax.set_ylabel(r'Average time (ms) per iteration', fontsize=16)
  ax.set_ylim(0.01, 1e5)

  ax.tick_params(axis='both', which='major', labelsize=12)
  ax.tick_params(axis='both', which='minor', labelsize=10)

  plt.tight_layout()
  filename = "fom_"+metric+"_"+stat+"_line.png"
  fig.savefig(filename, format="png", bbox_inches='tight', dpi=300)

#=====================================================================
def main(dataFile, metric, stat):
  data = np.loadtxt(dataFile)
  dataDic = createDataDic(data, metric, stat)
  print(dataDic)
  pp.pprint(dataDic)

  plotBar(dataDic, meshLabelsPlot, nThreads, metric, stat)
  plt.show()
  if (metric=="itertime"):
    plotLines(dataDic, meshLabelsPlot, nThreads, metric, stat)
    plt.show()

#////////////////////////////////////////////
if __name__== "__main__":
#////////////////////////////////////////////
  parser = ArgumentParser()
  # parser.add_argument("-file", "--file",
  #                     dest="dataFile",
  #                     help="where to get data from\n")

  parser.add_argument("-metric", "--metric",
                      dest="metric", default="mem",
                      help="Choices: mem, cpu, itertime \n")

  parser.add_argument("-stat", "--stat",
                      dest="stat", default="ave",
                      help="ave, min or max\n")

  args = parser.parse_args()

  assert(args.metric in ['mem', 'cpu', 'itertime'])
  main('./data/fom_scaling_final.txt', args.metric, args.stat)

#////////////////////////////////////////////
