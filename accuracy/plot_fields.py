#!/usr/bin/env python

import numpy as np
import sys, re, os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy import linalg as la

# we divide by 1000 to use kilometers
rLim=[2500000./1000, 7000000./1000.]
earthRad=6371000./1000
modelThickness=2891000./1000
# the cmb has a radius = earthRad - modelThickness
cmbRad=earthRad-modelThickness

def plotCMB(ax):
  # trace the CMB
  cmbTh = np.linspace(0, 2*np.pi, 100)
  cmbRa = cmbRad*np.ones(100)
  ax.plot(cmbTh, cmbRa, c='b', linestyle='--')

def plotEarthSurf(ax):
  # trace the earth surface
  surfTh = np.linspace(0, 2*np.pi, 100)
  surfRa = earthRad*np.ones(100)
  ax.plot(surfTh, surfRa, c='b')

#=========================================
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

#=========================================
def loadStates(fomFile, approxFile, dofName):
  # load states
  if not os.path.exists(fomFile):
    print("fom final state {} does not exist".format(fomFile))
    sys.exit(1)

  if not os.path.exists(approxFile):
    print("approx final state {} does not exist".format(approxFile))
    sys.exit(1)

  print("reading fom    state {}".format(fomFile))
  print("reading approx state {}".format(approxFile))

  # load data (skip first row because it contains the size)
  fomState              = np.loadtxt(fomFile, skiprows=1)
  fomStateReconstructed = np.loadtxt(approxFile, skiprows=1)
  fomNRows = fomState.shape[0]
  approxNRows = fomStateReconstructed.shape[0]
  # if things are correct, this should be a single vector and approx/fom match sizes
  assert( fomNRows == approxNRows )

  return [fomState, fomStateReconstructed]

#=========================================
def computeErrors(fomState, fomStateReconstructed, dofName):
  error = fomState-fomStateReconstructed
  print(" {}_fom_minmax    = {} {}".format(dofName, np.min(fomState), np.max(fomState)))
  print(" {}_approx_minmax = {} {}".format(dofName, np.min(fomStateReconstructed), np.max(fomStateReconstructed)))
  print(" {}_err_minmax    = {} {}".format(dofName, np.min(error), np.max(error)))
  fomL2Norm, fomLinfNorm = la.norm(fomState), la.norm(fomState, np.inf)
  errL2Norm   = [ la.norm(error),         la.norm(error)/fomL2Norm ]
  errLinfNorm = [ la.norm(error, np.inf), la.norm(error, np.inf)/fomLinfNorm ]
  print(" {}_err_abs_rel_ltwo_norms = {} {}".format(dofName, errL2Norm[0],   errL2Norm[1]))
  print(" {}_err_abs_rel_linf_norms = {} {}".format(dofName, errLinfNorm[0], errLinfNorm[1]))


def extractMeshSizeTheta(logFilePath, dof):
  reg = re.compile(r'nth = \d+')
  file1 = open(logFilePath, 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[2])

def extractMeshSizeRadius(logFilePath, dof):
  reg = re.compile(r'nr = \d+')
  file1 = open(logFilePath, 'r')
  strings = re.search(reg, file1.read())
  file1.close()
  assert(strings)
  return int(strings.group().split()[2])

###############################
if __name__== "__main__":
###############################
  # plot final velocity data for:
  # T=65 (reproductive)
  #   dir = fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_13

  # T=51.78003645 (middle of the sampling range)
  #   dir = fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_0

  # T=31 (extrapolation)
  #   dir = fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_10

  # load coordinates (which are the same for every case)
  nr, nth = 512, 2048
  cc = np.loadtxt("./data/coords_vp.txt")
  th, r = -cc[:,0]+np.pi/2., cc[:, 1]
  th, r = th.reshape((nr,nth)), r.reshape((nr,nth))

  Tvals = [65, 51.78003645, 31]

  fomFiles = ['./data/fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_13/finalFomState_vp_0',
              './data/fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_0/finalFomState_vp_0',
              './data/fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_10/finalFomState_vp_0']

  romFiles = ['./data/rom_mesh512x2048_nThreads_18_dt_0.1_T_2000.0_snaps_true_mat_prem_fRank_14_nPod_415_415/finalFomState_vp_13',
              './data/rom_mesh512x2048_nThreads_18_dt_0.1_T_2000.0_snaps_true_mat_prem_fRank_14_nPod_415_415/finalFomState_vp_0',
              './data/rom_mesh512x2048_nThreads_18_dt_0.1_T_2000.0_snaps_true_mat_prem_fRank_14_nPod_415_415/finalFomState_vp_10']

  interpFiles = ['./data/interpolation_n2/linear_test13.txt',
                 './data/interpolation_n2/linear_test0.txt',
                 './data/interpolation_n2/linear_test10.txt']

  cm = plt.cm.get_cmap('PuOr') #BrBG_r')
  for T,fom,rom,interp in zip(Tvals[0:1], fomFiles[0:1], romFiles[0:1], interpFiles[0:1]):
    print(T,fom,rom)
    fomState = np.loadtxt(fom, skiprows=1)
    romState = np.loadtxt(rom, skiprows=1)
    computeErrors(fomState, romState, "vp")
    error = fomState-romState


    fig1 = plt.figure(0)
    ax1 = fig1.add_subplot(111, projection='polar')
    h1=ax1.pcolormesh(th, r, fomState.reshape((nr,nth)), cmap=cm, shading = "flat",
                      vmin=-8e-10, vmax=8e-10)
    ax1.set_rlabel_position(260)
    fig1.colorbar(h1)
    plt.show()





  # [fomState, fomStateReconstructed] = loadStates(args.fomState, args.approxState, "vp")
  # computeErrors(fomState, fomStateReconstructed, "vp")
  # error = fomState-fomStateReconstructed

  # # should find from log file inside fom dir the mesh size

  # cc = np.loadtxt("./coords_vp.txt")

  # cm = plt.cm.get_cmap('PuOr') #BrBG_r')

  # fig1 = plt.figure(1)
  # ax1 = fig1.add_subplot(111, projection='polar')
  # h1=ax1.pcolormesh(th, r, fomState.reshape((nr,nth)),
  #                   cmap=cm, shading = "flat",
  #                   vmin=-8e-10, vmax=8e-10)
  # ax1.set_rlabel_position(260)
  # fig1.colorbar(h1)

  # fig2 = plt.figure(2)
  # ax2 = fig2.add_subplot(111, projection='polar')
  # h2=ax2.pcolormesh(th, r, fomStateReconstructed.reshape((nr,nth)),
  #                   cmap=cm, shading = "flat",
  #                   vmin=-8e-10, vmax=8e-10)
  # ax2.set_rlabel_position(260)
  # fig2.colorbar(h2)

  # fig3 = plt.figure(3)
  # ax3 = fig3.add_subplot(111, projection='polar')
  # h3=ax3.pcolormesh(th, r, error.reshape((nr,nth)),
  #                   cmap="binary", shading = "flat",
  #                   vmin=-8e-10, vmax=8e-10)
  # ax3.set_rlabel_position(260)
  # fig3.colorbar(h3)

  # plt.show()


  # # com = plt.cm.get_cmap('PuOr') #BrBG_r')
  # # ax.pcolormesh(th, r, z, cmap=com, shading = "flat",
  # #               alpha=1, vmin=-8e-10, vmax=8e-10)
  # # #fig.savefig("snap.png", format="png", bbox_inches='tight', dpi=300)
