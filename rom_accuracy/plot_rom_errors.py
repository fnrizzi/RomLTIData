#!/usr/bin/env python

from random import randrange, uniform
import re, sys, os, time, yaml, copy
import numpy as np
from argparse import ArgumentParser
import shutil, subprocess
import matplotlib.pyplot as plt

from find_train_points import *

np.set_printoptions(linewidth=100)

#=========================================
def findTrainPoints(workDir, scenario):
  # get all fom train dirs
  fomDirsFullPath = [workDir+'/'+d for d in os.listdir(workDir) if 'fom' in d and 'train' in d]
  # sort based on the test ID
  def func(elem): return int(elem.split('_')[-1])
  fomDirsFullPath = sorted(fomDirsFullPath,key=func)

  data = np.zeros(len(fomDirsFullPath))
  for i, idir in enumerate(fomDirsFullPath):
    ifile = idir + '/input.yaml'
    inputs = yaml.safe_load(open(ifile))
    if scenario==1: data[i] = inputs['material']['layer2']['velocity'][0]
    if scenario==2: data[i] = inputs['source']['signal']['period']
  return data

#=========================================
def doPlot(trainVals, M, dof, scenario, normKind, workDir):
  if scenario==1:
    # 28 117 222 1117 1914 2227 2329 2385
    romSizes = [117, 222, 1117, 2329, 2385]
    mk = {117:'D', 222:'p', 1117:'>', 2329:'v', 2385:'o'}
  elif scenario==2:
    # 170, 275, 311, 342, 369, 393, 415, 436
    romSizes = [311, 369, 415, 436]
    mk = {311:'D', 369:'p', 415:'>', 436:'o'}
  print(romSizes)

  # col indices where to find the errors
  # for ROM, the data is an array where:
  #             col0   : testValue
  #             col1   : rom size
  #             col2,3 : abs-l2 and rel-l2
  #             col4,5 : abs-linf and rel-linf
  absL2,   relL2   = 2, 3
  absLInf, relLInf = 4, 5

  if normKind==-1:
    targetCol = relLInf
    normStr = 'linf'
    ylabnrm = '\ell_{\infty}'
  if normKind==2:
    targetCol = relL2
    normStr = 'l2'
    ylabnrm = '\ell_{2}'

  # extract data for a given rom size
  fig = plt.figure(0)
  ax = plt.gca()
  for romS in romSizes:
    # select only where we match the current romsize
    D = M[ M[:,1] == romS]
    # sort by the the test points so that line plots works ok
    D = D[D[:,0].argsort()]
    plt.plot(D[:,0], D[:, targetCol], '-', marker=mk[romS],
             #markerfacecolor='none',
             markersize=7, linewidth=1.5, label='p='+str(int(romS)))

  for x in trainVals:
    midTrain = 0.5*(trainVals[0]+trainVals[1])
    ax.annotate('Train points', xy=(x, 0.0), xytext=(midTrain, 0.5), size=16,
                arrowprops=dict(facecolor='black', shrink=10,
                                headwidth=7, width=0.7, linewidth=0.5),
                horizontalalignment='center')

  ax.legend(loc="upper right", ncol=1, fontsize=13, frameon=False, labelspacing=0.2, handletextpad=0.01)

  if dof=='vp':  ylabDof = 'velocity'
  else: ylabDof = 'stresses'
  ax.set_ylabel(r'E$_{'+ylabnrm+'}$ for ' + ylabDof, fontsize=15)

  if scenario==1:
    ax.set_xlabel('Shear velocity (km/s)', fontsize=16)
    ax.set_xlim(5900, 6400)
    # ax.set_xticks(np.linspace(30, 70, 9))
  elif scenario==2:
    ax.set_xlabel('Forcing period (sec)', fontsize=16)
    ax.set_xlim(28, 72)
    ax.set_xticks(np.linspace(30, 70, 9))
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  ax.set_yticks(np.linspace(0, 1, 11))
  plt.grid()

  fileName = 'rom_acc_sce_'+str(scenario)+'_errors_'+dof+'_'+normStr+'.pdf'
  plotDir = workDir+'/plots'
  if not os.path.exists(plotDir): os.system('mkdir -p ' + plotDir)
  fig.savefig(plotDir+'/'+fileName, format="pdf", bbox_inches='tight', dpi=300)
  plt.show()

###############################
if __name__== "__main__":
###############################
  parser = ArgumentParser()
  parser.add_argument("-wdir", "--wdir",
                      dest="workDir", default="empty",
                      help="Target dir such that I can find workDir/parsed_data. Must be set.")

  parser.add_argument("-scenario", "--scenario",
                      dest="scenario", default=0, type=int,
                      help="Choices: 1 (uncertain velocity), 2 (uncertain forcing period, fixed delay). Must set.")

  parser.add_argument("-norm", "--norm",
                      dest="normKind", default=2, type=int,
                      help="Choices: -1 (linf-norm), 2 (l2-norm).")

  #------------------------------
  args = parser.parse_args()
  assert(args.workDir != "empty")
  assert(args.scenario in [1,2])
  assert(args.normKind in [-1,2])
  workDir  = args.workDir
  scenario = args.scenario
  nrm = args.normKind

  parsedDataDir = workDir+'/parsed_data'
  dataDir       = workDir+'/data'

  # find the values used for pod training
  trainVals = findTrainPoints(dataDir, scenario)
  print("trainValues = {}".format(trainVals))

  for dof in ['vp', 'sp']:
    data = np.loadtxt(parsedDataDir+'/rom_errors_table_'+dof+'.txt')
    doPlot(trainVals, data, dof, scenario, nrm, workDir)
