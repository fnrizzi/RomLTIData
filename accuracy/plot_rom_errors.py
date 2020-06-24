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
    if scenario==2:
      data[i] = inputs['source']['signal']['period']
  return data


#=========================================
def doPlot(trainVals, M, dof, scenario, normKind):
  romSizes = [311, 369, 415, 436]
  print(romSizes)

  mk = {311:'D', 369:'p', 415:'>', 436:'o'}

  # # find all unique test points (unique sorts by default, so fix that)
  # testPts = M[:,0]
  # indexes = np.unique(testPts, return_index=True)[1]
  # testPts = [testPts[index] for index in sorted(indexes)]
  # print(testPts)

  # col indices where to find the errors
  # for ROM, the data is an array where:
  #             col0   : testValue
  #             col1   : rom size
  #             col2,3 : abs-l2 and rel-l2
  #             col4,5 : abs-linf and rel-linf
  absL2, relL2 = 2, 3
  absLInf, relLInf = 4, 5

  if normKind==-1:
    targetCol = relLInf
    normStr = 'linf'
    ylabnrm = '\ell_{\infty}'
  if normKind==2:
    targetCol = relL2
    normStr = 'l2'
    ylabnrm = '\ell_{2}'

  # extract data from M for a given rom size
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
    ax.annotate('Train pts for \nPOD basis', xy=(x, 0.0),
                xytext=(50, 0.5), size=16,
                arrowprops=dict(facecolor='black', shrink=10,
                                headwidth=7, width=0.7, linewidth=0.5),
                horizontalalignment='center')
    #plt.plot([x, x], [0,15], '--k', linewidth=1.5)

  ax.set_xlim(28, 72)
  ax.set_ylim(0, 1.)
  ax.legend(loc="upper right", ncol=1, fontsize=13,
            frameon=False, labelspacing=0.2, handletextpad=0.01)

  #ax.set_yscale('log')
  ax.set_xlabel('Forcing period (sec)', fontsize=16)
  if dof=='vp':  ylabDof = 'velocity'
  else: ylabDof = 'stresses'

  ax.set_ylabel(r'E$_{'+ylabnrm+'}$ for ' + ylabDof, fontsize=15)

  ax.set_xticks(np.linspace(30, 70, 9))
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  ax.set_yticks(np.linspace(0, 1, 11))
  plt.grid()

  fileName = 'rom_acc_sce_'+str(scenario)+'_errors_'+dof+'_'+normStr+'.pdf'
  fig.savefig('./plots/'+fileName, format="pdf", bbox_inches='tight', dpi=300)
  plt.show()

###############################
if __name__== "__main__":
###############################
  parser = ArgumentParser()
  parser.add_argument("-scenario", "--scenario",
                      dest="scenario", default=0, type=int,
                      help="Choices: 1 (uncertain velocity), 2 (uncertain forcing period, fixed delay). Must set.")

  parser.add_argument("-norm", "--norm",
                      dest="normKind", default=2, type=int,
                      help="Choices: -1 (linf-norm), 2 (l2-norm).")

  #------------------------------
  args = parser.parse_args()
  assert(args.scenario in [1,2])
  assert(args.normKind in [-1,2])
  scenario = args.scenario
  nrm = args.normKind

  # find the values used for pod training
  trainVals = findTrainPoints('./data', scenario)
  print("trainValues = {}".format(trainVals))

  for dof in ['vp', 'sp']:
    data = np.loadtxt('./parsed_data/rom_errors_table_'+dof+'.txt')
    doPlot(trainVals, data, dof, scenario, nrm)
