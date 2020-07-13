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
def doPlot(trainVals, dataDic, dof, scenario, normKind, workDir):
  mk = ['o', 's', '*']
  color = ['k', 'r', 'b']

  # col indices where to find the errors
  #             col0   : testValue
  #             col1   : numInterpPts
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
  # sort by the the test points so that line plots works ok
  i = 0
  for key,M in dataDic.items():
    M = M[M[:,0].argsort()]
    plt.plot(M[:,0], M[:, targetCol], '-', marker=mk[i],
             color=color[i], markerfacecolor=color[i],
             markersize=7, linewidth=1.5, label=key)
    i+=1

  # for x in trainVals:
  #   textXY = [65., 0.6] if scenario==2 else [6050., 0.8]
  #   ax.annotate('Train pts', xy=(x, 0.02), xytext=(textXY[0], textXY[1]), 
  #               size=16, 
  #               arrowprops=dict(facecolor='black', shrink=10, linestyle=':',
  #                               headwidth=7, width=0.7, linewidth=0.5),
  #               horizontalalignment='center')

  ax.legend(loc="upper right", ncol=1, fontsize=12, frameon=False, labelspacing=0.1, handletextpad=0.01)

  if dof=='vp':  ylabDof = 'velocity'
  else: ylabDof = 'stresses'
  ax.set_ylabel(r'E$_{'+ylabnrm+'}$ for ' + ylabDof, fontsize=15)

  if scenario==1:
    ax.set_xlabel('Shear velocity (km/s)', fontsize=16)
    ax.set_xlim(5900, 6400)
    ax.set_xticks(np.linspace(5900, 6400, 6))
  elif scenario==2:
    ax.set_xlabel('Forcing period (sec)', fontsize=16)
    ax.set_xlim(28, 72)
    ax.set_xticks(np.linspace(30, 70, 9))
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  ax.set_yticks(np.linspace(0, 1, 11))
  plt.grid()

  fileName = 'interp_acc_sce_'+str(scenario)+'_errors_'+dof+'_'+normStr+'.pdf'
  fig.savefig(workDir+'/plots/'+fileName, format="pdf", bbox_inches='tight', dpi=300)
  plt.show()

###############################
if __name__== "__main__":
###############################
  parser = ArgumentParser()
  parser.add_argument("-wdir", "--wdir",
                      dest="workDir", default="empty",
                      help="Target dir such that I can find data. Must be set.")

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
  workDir  = args.workDir
  scenario = args.scenario
  nrm = args.normKind

  parsedDataDir = workDir+'/parsed_data'
  dataDir       = workDir+'/data'

  # find the values used for training
  trainVals = findTrainPoints(dataDir, scenario)
  print("trainValues = {}".format(trainVals))
  n = len(trainVals)

  for dof in ['vp', 'sp']:
    dataNN  = np.loadtxt(parsedDataDir+'/interp_n'+str(n)+'_errors_table_'+dof+'_nn.txt');
    dataLin = np.loadtxt(parsedDataDir+'/interp_n'+str(n)+'_errors_table_'+dof+'_linear.txt');
    dataDic = {'Nearest neighbors': dataNN, 'Linear': dataLin}
    if n>2:
      dataQ = np.loadtxt(parsedDataDir+'/interp_n'+str(n)+'_errors_table_'+dof+'_quadratic.txt');
      dataDic['Quadratic'] = dataQ

    doPlot(trainVals, dataDic, dof, scenario, nrm, workDir)
