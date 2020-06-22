#!/usr/bin/env python

from random import randrange, uniform
import re, sys, os, time, yaml, copy
import numpy as np
from argparse import ArgumentParser
import shutil, subprocess
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100)

#=========================================
def getRunIDFromDirName(dir):
  return int(dir.split('_')[-1])

#=========================================
def extractErrors(logFilePath, dof, kind):
  if dof == 'vp' and kind == 'l2':
    reg = re.compile(r'vp_err_abs_rel_ltwo_norms = .+')
  elif dof == 'vp' and kind == 'linf':
    reg = re.compile(r'vp_err_abs_rel_linf_norms = .+')
  if dof == 'sp' and kind == 'l2':
    reg = re.compile(r'sp_err_abs_rel_ltwo_norms = .+')
  elif dof == 'sp' and kind == 'linf':
    reg = re.compile(r'sp_err_abs_rel_linf_norms = .+')

  file1 = open(logFilePath, 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return np.array([float(i) for i in strings.group().split()[2:]])

#=========================================
def extractParamValue(scenario, tdir):
  # the input file
  ifile = tdir + '/input.yaml'
  inputs = yaml.safe_load(open(ifile))
  if scenario==2:
    return inputs['source']['signal']['period']

#=========================================
def extractSamplingValues(tdir):
  # the input file
  ifile = tdir + '/input.yaml'
  inputs = yaml.safe_load(open(ifile))
  return inputs['sampling']['values']

#=========================================
def extractRomSize(tdir):
  ifile = tdir + '/input.yaml'
  inputs = yaml.safe_load(open(ifile))
  vpRomSize = inputs['rom']['velocity']['numModes']
  spRomSize = inputs['rom']['stress']['numModes']
  if (vpRomSize == spRomSize):
    return vpRomSize
  else:
    sys.exit('num of modes used for velocity and stress is different')

#=========================================
def parseRomErrors(scenario, workDir, dofName):
  assert(scenario==2)
  # in this case we have fom tests enumrated as test_0, test_1, etc..
  # but we have one ROM dir for each # of modes that contains all samples of the forcing
  # inside each rom test dir, we have one file for each test as:
  # for instante, for numModes = X, we have a dir:
  #     rom_nThreads_..._nPod_X_X
  # inside we have:
  # finalFomState_{vp,sp}_i: final fom state for i-th test

  # get all fom test dirs
  fomDirsFullPath = [workDir+'/'+d for d in os.listdir(workDir) if 'fom' in d and 'test' in d]
  # sort based on the test ID
  def func(elem): return int(elem.split('_')[-1])
  fomDirsFullPath = sorted(fomDirsFullPath,key=func)

  # get all rom dirs
  romDirsFullPath = [workDir+'/'+d for d in os.listdir(workDir) if 'rom' in d]
  # sort based on the rom size (which is at end of dir name)
  def func(elem): return int(elem.split('_')[-1])
  romDirsFullPath = sorted(romDirsFullPath,key=func)

  numTestPts = len(fomDirsFullPath)
  print(numTestPts)
  numRomSizes = len(romDirsFullPath)
  nR = numTestPts * numRomSizes
  nC = 6
  data = np.zeros((nR, nC))

  # loop over fom test dirs
  r = 0
  for fomTestDir in fomDirsFullPath:
    print('fomTestDir: {}'.format(fomTestDir))

    # extract the test ID of this run
    testID = getRunIDFromDirName(fomTestDir)
    # get the param value of this test (depends on scenario)
    thisTestValue = extractParamValue(scenario, fomTestDir)
    print(thisTestValue)

    for romDir in romDirsFullPath:
      # rom file to extract errors
      romErrFile = romDir + '/finalError_' + dofName + '_' + str(testID) + '.txt'
      ltwo = extractErrors(romErrFile, dofName, 'l2')
      linf = extractErrors(romErrFile, dofName, 'linf')

      # make sure that the test value we are dealing with is found at the right index
      # in the samples run by rom match
      samples = extractSamplingValues(romDir)
      assert( thisTestValue == samples[testID] )

      data[r,0]   = thisTestValue
      data[r,1]   = extractRomSize(romDir)
      data[r,2:4] = ltwo
      data[r,4:6] = linf
      r+=1
  return data

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
def doPlot(trainVals, M, dof):
  romSizes = [170, 311, 369, 415, 436] # np.unique(M[:,1])
  print(romSizes)

  mk = {170:'s', 311:'D', 369:'p', 415:'>', 436:'o'}

  # find all unique test points
  testPts = np.unique(M[:,0])
  print(testPts)

  absL2, relL2 = 2, 3
  absLInf, relLInf = 4, 5

  # extract data from M for a given rom size
  fig = plt.figure(0)
  ax = plt.gca()
  for romS in romSizes:
    D = M[ M[:,1] == romS]
    plt.plot(testPts, D[:,relL2], '-', marker=mk[romS], #markerfacecolor='none',
             markersize=7, linewidth=1.5, label='p='+str(int(romS)))

  for x in trainVals:
    ax.annotate('POD train pts', xy=(x, 7), xytext=(trainVals[0], 30), size=16,
                arrowprops=dict(facecolor='black', shrink=10, headwidth=7, width=0.7, linewidth=0.5),
                horizontalalignment='center')
    plt.plot([x, x], [0,7], '--k', linewidth=1.5)

  ax.set_xlim(28, 72)
  ax.set_ylim(1e-4, 300)
  ax.legend(loc="upper right", ncol=2, fontsize=13, frameon=False, labelspacing=0.2,
            handletextpad=0.01)

  ax.set_yscale('log')
  ax.set_xlabel('Forcing period (sec)', fontsize=16)
  if dof=='vp':
    ax.set_ylabel(r'Relative $\ell_2$-norm of velocity error', fontsize=15)
  else:
    ax.set_ylabel(r'Relative $\ell_2$-norm of stress error', fontsize=15)
  ax.set_xticks(np.linspace(30, 70, 9))
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  ax.set_yticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0])
  plt.grid()

  fig.savefig('errors_'+dof+'.pdf', format="pdf", bbox_inches='tight', dpi=300)
  plt.show()


###############################
if __name__== "__main__":
###############################
  parser = ArgumentParser()

  parser.add_argument("-working-dir", "--working-dir", "-wdir", "--wdir",
                      dest="workDir", default="empty",
                      help="Target dir where to work. Must be set.")

  parser.add_argument("-scenario", "--scenario",
                      dest="scenario", default=0, type=int,
                      help="Choices: 1 (uncertain velocity), 2 (uncertain forcing period, fixed delay). Must set.")

  #------------------------------
  # parse all args
  #------------------------------
  args = parser.parse_args()
  assert(args.workDir != "empty")
  assert(args.scenario in [1,2])
  workDir  = args.workDir
  scenario = args.scenario

  # check if working dir exists, if not, make it
  if not os.path.exists(workDir):
    sys.exit("The workdir {} provided does not exist".format(args.exeDir))

  # find the values used for pod training
  trainVals = findTrainPoints(workDir, scenario)
  print(trainVals)

  # # data is an array where:
  # # col0   : testValue
  # # col1   : rom size
  # # col2,3 : abs-l2 and rel-l2
  # # col4,5 : abs-linf and rel-linf
  #data = parseRomErrors(scenario, workDir, 'vp'); np.savetxt('errors_table_vp.txt', data)
  #data = parseRomErrors(scenario, workDir, 'sp'); np.savetxt('errors_table_sp.txt', data)
  data = np.loadtxt('errors_table_vp.txt'); doPlot(trainVals, data, 'vp')
  data = np.loadtxt('errors_table_sp.txt'); doPlot(trainVals, data, 'sp')
