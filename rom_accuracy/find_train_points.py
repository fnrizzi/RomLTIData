#!/usr/bin/env python

from random import randrange, uniform
import re, sys, os, time, yaml, copy
import numpy as np
from argparse import ArgumentParser
import shutil, subprocess

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
    if scenario==1:
      data[i] = inputs['material']['layer2']['velocity'][0]
    if scenario==2:
      data[i] = inputs['source']['signal']['period']
  return data
