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
def extractErrors(logFilePath, dof, normkind, testID, kind):
  if normkind == 'l2':
    reg = re.compile(r''+kind+'_test'+str(testID)+'_'+dof+'_err_\D+_ltwo_\D+ = .+')
  elif normkind == 'linf':
    reg = re.compile(r''+kind+'_test'+str(testID)+'_'+dof+'_err_\D+_linf_\D+ = .+')

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
def parseErrors(scenario, workDir, dofName, kind, numInterpPts):
  interpDataDir = workDir+'/interpolation_n'+str(numInterpPts)

  # get all fom test dirs
  fomDirsFullPath = [workDir+'/'+d for d in os.listdir(workDir) if 'fom' in d and 'test' in d]
  # sort based on the test ID
  def func(elem): return int(elem.split('_')[-1])
  fomDirsFullPath = sorted(fomDirsFullPath,key=func)

  numTestPts = len(fomDirsFullPath)
  print(numTestPts)
  nR = numTestPts
  data = np.zeros((nR, 6))

  # the file with all interpolation errors
  interpLogFile = interpDataDir+'/interpolation_'+dofName+'.log'
  print(interpLogFile)

  # loop over fom test dirs
  r = 0
  for fomTestDir in fomDirsFullPath:
    print('fomTestDir: {}'.format(fomTestDir))

    # extract the test ID of this run
    testID = getRunIDFromDirName(fomTestDir)
    # get the param value of this test (depends on scenario)
    thisTestValue = extractParamValue(scenario, fomTestDir)
    print(thisTestValue)

    # extract l2 errors
    ltwo = extractErrors(interpLogFile, dofName, 'l2', testID, kind)
    linf = extractErrors(interpLogFile, dofName, 'linf', testID, kind)

    data[r,0]   = thisTestValue
    data[r,1]   = numInterpPts
    data[r,2:4] = ltwo
    data[r,4:6] = linf
    r+=1
  return data

###############################
if __name__== "__main__":
###############################
  parser = ArgumentParser()
  parser.add_argument("-wdir", "--wdir",
                      dest="workDir", default="empty",
                      help="Target dir with the data.")

  parser.add_argument("-scenario", "--scenario",
                      dest="scenario", default=0, type=int,
                      help="Choices: 1 (uncertain velocity), 2 (uncertain forcing period, fixed delay).")

  #------------------------------
  # parse all args
  #------------------------------
  args = parser.parse_args()
  assert(args.workDir != "empty")
  assert(args.scenario in [1,2])
  workDir  = args.workDir
  scenario = args.scenario

  parsedDataDir = workDir+'/parsed_data'
  if not os.path.exists(parsedDataDir):
    os.system('mkdir -p ' + parsedDataDir)

  dataDir       = workDir+'/data'

  # find the values used for training
  trainVals = findTrainPoints(dataDir, scenario)
  print("trainValues = {}".format(trainVals))
  n = len(trainVals)

  interpolants = ['nn', 'linear'] if n == 2 else ['nn', 'linear', 'quadratic']
  for dof in ['vp', 'sp']:
    for kind in interpolants:
      data = parseErrors(scenario, dataDir, dof, kind, n)
      np.savetxt(parsedDataDir+'/interp_n'+str(n)+'_errors_table_'+dof+'_'+kind+'.txt', data)
