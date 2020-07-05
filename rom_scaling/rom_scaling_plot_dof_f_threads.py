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

# peak BW for mrstem
# (GB) = 21333.33 * 4 at time of doing these runs
peakMemBW = 85.

myAlpha = 0.9

modes    = [256, 512, 1024, 2048, 4096]

nThreads = [1, 8, 36]
colors={1:'#2d4676', 8:'#ff9e11', 36:'#cd4d84'}

fSizes   = [1, 16, 128, 512, 1024]

# colors   = {1:    '#003f5c',
#             16:   '#ffa600',
#             128:  'gray',
#             512:  '#7a5195',
#             1024: '#ff6361'} ##00818a'}

#=====================================================================
def computeMetricValue(lineData, currValueF, nThr, metric, stat):
  if metric == "mem":
    if stat == "ave": return lineData[5]
    elif stat=="min": return lineData[6]
    elif stat=="max": return lineData[7]

  elif metric == "cpu":
    if stat == "ave": return lineData[8]
    elif stat=="min": return lineData[9]
    elif stat=="max": return lineData[10]

  elif metric == "itertime":
    if stat == "ave": return lineData[11]
    elif stat=="min": return lineData[12]
    elif stat=="max": return lineData[13]

  elif metric == "looptime":
    return lineData[14]

#=====================================================================
def createDataDic(data, metric, stat):
  all = {}
  for nt in nThreads:
    dic = {}
    for i in range(data.shape[0]):
      # number of threads and number of modes
      thisNumThr   = int(data[i][0])
      thisValF     = int(data[i][1])
      thisNumModes = int(data[i][2])

      if thisNumThr == nt and thisValF in fSizes and thisNumModes in modes:
          value = computeMetricValue(data[i,:], thisValF, nt, metric, stat)

          if thisValF in dic: dic[thisValF].append(value)
          else: dic[thisValF] = [value]
    all[nt] = dic

  return all

#=====================================================================
def plotBarSet(ax, xLoc, width, f, dic, myColor):
  val = dic[f]
  ax.bar(xLoc, val, width, alpha=myAlpha, color=myColor, edgecolor='none', zorder=5)

#=====================================================================
def plotBar(dataDic, nThreads, metric, stat):
  # number of modes to deal with
  numModes = len(modes)
  # Setting the positions and width for the bars
  posArray = range(numModes)
  pos = list(posArray)

  width = 0.375

  plt.rc('axes', axisbelow=True)

  fig, ax = plt.subplots(figsize=(9,6))
  plt.grid()
  ax2 = ax.twiny()
  fig.subplots_adjust(bottom=0.25)

  gigi = [0.25, 6.5, 12.75, 19., 25.25]

  #---- loop over numthreads and plot ----
  xTicksBars, xTlabels = [], []
  count=0
  for k,v in dataDic.items():
    for i,f in enumerate(fSizes):
      #x locations for the bars
      shift = width*i*3.4

      xLoc = [p+shift+width*count+gigi[k] for k,p in enumerate(pos)]

      plotBarSet(ax, xLoc, width, f, v, colors[k])
      xTicksBars += [p+shift+0.35+gigi[k] for k,p in enumerate(pos)]
      xTlabels += [str(f) for i in range(numModes)]
    count+=1

  for nt in nThreads:
    ax.bar(100, 1, width, alpha=myAlpha, color=colors[nt],
           edgecolor='none', zorder=-1, label='nThr='+str(nt))

  ax.legend(loc="upper left", ncol=5, fontsize=13, frameon=False)

  # vertical lines of the grid
  ax.xaxis.grid(which="major", color='None', linestyle='-.', linewidth=0, zorder=0)

  ax.xaxis.set_ticks_position('bottom')
  ax.xaxis.set_label_position('bottom')
  ax.set_xticks(xTicksBars)
  ax.set_xticklabels(xTlabels, fontsize=13)
  ax.xaxis.set_tick_params(rotation=35)

  ax.set_xlabel(r'Size of f', fontsize=15)
  ax.set_xlim(min(pos)-0.2, max(pos)+width*84)

  if metric =="mem":
    ax.set_yscale('log')
    ax.set_ylabel("Memory Bandwith (GB/s)", fontsize=18)
    ax.set_ylim([1e-1, 1000])
    ax.set_yticks([1e-1, 1, 10, 100, 1000])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)

  elif metric =="cpu":
    ax.set_yscale('log')
    ax.set_ylabel("GFlops", fontsize=18)
    ax.set_ylim([1e-1, 1e4])
    ax.set_yticks([1e-1, 1, 10, 1e2, 1e3, 1e4])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)

  elif metric =="itertime":
    ax.set_yscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1e-1, 1e4])
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='y', which='minor', labelsize=13)
    if stat == 'ave': pref = 'Average'
    elif stat=='min': pref = 'Min'
    elif stat=='max': pref = 'Max'
    ax.set_ylabel(pref+" time (ms)/timestep", fontsize=18)

  # ticks for the rom sizes
  meshTicks = [3.4, 10.5, 17.75, 25, 32.1]
  ax2.set_xticks(meshTicks)
  ax2.xaxis.set_ticks_position('bottom')
  ax2.xaxis.set_label_position('bottom')
  ax2.spines['bottom'].set_position(('outward', 65))
  ax2.set_xlabel('ROM Size', fontsize=16)
  ax2.set_xticklabels([r''+str(m)+'' for m in modes], fontsize=16)
  ax2.set_xlim(min(pos), max(pos)+width*84)
  ax2.set_axisbelow(True)

  plt.tight_layout()
  filename = "rom_"+metric+"_"+stat+".png"
  fig.savefig('./plots/'+filename, format="png", bbox_inches='tight', dpi=300)

#=====================================================================
def main(dataFile, metric, stat):
  data = np.loadtxt(dataFile)
  dataDic = createDataDic(data, metric, stat)
  pp.pprint(dataDic)

  plotBar(dataDic, nThreads, metric, stat)
  plt.show()
  #if (metric in ['itertime']): plotLines(dataDic, nThreads, metric, stat)

#////////////////////////////////////////////
if __name__== "__main__":
#////////////////////////////////////////////
  parser = ArgumentParser()
  # parser.add_argument("-file", "--file",
  #                     dest="dataFile",
  #                     help="where to get data from\n")

  parser.add_argument("-metric", "--metric",
                      dest="metric", default="mem",
                      help="Choices: mem, cpu, looptime, itertime\n")

  parser.add_argument("-stat", "--stat",
                      dest="stat", default="ave",
                      help="ave, min or max\n")

  args = parser.parse_args()
  assert(args.metric in ['mem', 'cpu', 'itertime'])

  main('./data/rom_scaling_final.txt', args.metric, args.stat)
