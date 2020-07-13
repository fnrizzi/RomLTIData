#!/usr/bin/env python

from random import randrange, uniform
import re, sys, os, time, yaml, copy
import numpy as np
from argparse import ArgumentParser
import shutil, subprocess

from find_train_points import *

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
  elif dof == 'sp' and kind == 'l2':
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
  if scenario==1:
    return inputs['material']['layer2']['velocity'][0]
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
def parseRomErrorsScenario1(dataDir, dofName):
  # scenario1 is the one where we have a bilayer model,
  # and we sample the shear velocity in the second layer

  # we have fom tests enumrated as test_0, test_1, etc..
  # in this case we cannot use the multi-rank trick, so for each # of modes,
  # and each test case, we have a ROM dir
  #       rom_mesh_..._nThreads_..._nPod_X_X_test_i for the i-th test

  # get all fom test dirs
  fomDirsFullPath = [dataDir+'/'+d for d in os.listdir(dataDir) if 'fom' in d and 'test' in d]
  # get all rom dirs
  romDirsFullPath0 = [dataDir+'/'+d for d in os.listdir(dataDir) if 'rom' in d]

  # compute tota num of modes cases since we know we have one dir for each mode
  numTestPts = len(fomDirsFullPath)
  assert( len(romDirsFullPath0) % numTestPts == 0)
  numRomSizes = int(len(romDirsFullPath0)/numTestPts)
  print( numRomSizes)

  # first, sort based on the test ID
  def func(elem): return int(elem.split('_')[-1])
  romDirsFullPath1 = sorted(romDirsFullPath0,key=func)
  print(romDirsFullPath1[0:8])
  # then, for each test id, sort based on the rom size
  def func2(elem): return int(elem.split('_')[-3])
  romDirsFullPath = []
  for i in range(0, numTestPts):
    start = i*numRomSizes
    stop  = start+numRomSizes
    currSublist = romDirsFullPath1[start:stop]
    romDirsFullPath += sorted(currSublist, key=func2)

  for i in range(0, numTestPts):
    print(romDirsFullPath[i*numRomSizes])

  nR = len(romDirsFullPath)
  nC = 6
  data = np.zeros((nR, nC))

  # loop over rom dirs
  r = 0
  for romDir in romDirsFullPath:
    print('romTestDir: {}'.format(romDir))
    # extract the test ID of this run
    testID = getRunIDFromDirName(romDir)

    # get the param value of this test (depends on scenario)
    thisTestValue = extractParamValue(1, romDir)
    print(thisTestValue)

    # rom file to extract errors
    romErrFile = romDir + '/finalError_' + dofName + '.txt'
    ltwo = extractErrors(romErrFile, dofName, 'l2')
    linf = extractErrors(romErrFile, dofName, 'linf')

    data[r,0]   = thisTestValue
    data[r,1]   = extractRomSize(romDir)
    data[r,2:4] = ltwo
    data[r,4:6] = linf
    r+=1
  return data

#=========================================
def parseRomErrorsScenario2(dataDir, dofName):
  # scenario2 is the one where we sample the forcing period

  # in this case we have fom tests enumrated as test_0, test_1, etc..
  # but we have, for each # of modes, one single ROM dir containing
  # all samples of the forcing since we use the multiple rhs trick.
  # we have one file for each test as:
  # for numModes = X, we have a dir:
  #       rom_mesh_..._nThreads_..._nPod_X_X
  # inside we have:
  #       finalError_{vp,sp}_i: final fom state for i-th test

  # get all fom test dirs
  fomDirsFullPath = [dataDir+'/'+d for d in os.listdir(dataDir) if 'fom' in d and 'test' in d]
  # sort based on the test ID
  def func(elem): return int(elem.split('_')[-1])
  fomDirsFullPath = sorted(fomDirsFullPath,key=func)

  # get all rom dirs
  romDirsFullPath = [dataDir+'/'+d for d in os.listdir(dataDir) if 'rom' in d]
  print(romDirsFullPath)
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
    thisTestValue = extractParamValue(2, fomTestDir)
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
def parseRomErrors(scenario, dataDir, dofName):
  if scenario==1:
    return parseRomErrorsScenario1(dataDir, dofName)
  elif scenario==2:
    return parseRomErrorsScenario2(dataDir, dofName)
  else:
    sys.exit('invalid scenario')

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

  #------------------------------
  # parse all args
  #------------------------------
  args = parser.parse_args()
  assert(args.workDir != "empty")
  assert(args.scenario in [1,2])
  workDir  = args.workDir
  scenario = args.scenario

  dataDir = workDir+'/data'
  dataVp = parseRomErrors(scenario, dataDir, 'vp')
  dataSp = parseRomErrors(scenario, dataDir, 'sp')

  # check if dir where to put parsed data exists, if not, make it
  parsedDataDir = workDir+'/parsed_data'
  if not os.path.exists(parsedDataDir):
    os.system('mkdir -p ' + parsedDataDir)
  np.savetxt(parsedDataDir+'/rom_errors_table_vp.txt', dataVp)
  np.savetxt(parsedDataDir+'/rom_errors_table_sp.txt', dataSp)
