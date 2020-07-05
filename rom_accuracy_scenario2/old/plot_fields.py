#!/usr/bin/env python

import numpy as np
import sys, re, os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy import linalg as la
from matplotlib.patches import Circle, PathPatch

# radius of the earth
earthRadius = 6371. #km
# thickness of the mantle
mantleThickness = 2891. # km
# the radius of the core-mantle boundary
cmbRadius = earthRadius - mantleThickness

transition=5701.

def plotTransitionBD(ax):
  cmbTh = np.linspace(0, 2*np.pi, 100)
  ax.plot(cmbTh, transition*np.ones(100), c='k', linestyle='--', linewidth=0.25, zorder=2)

def plotCMB(ax):
  # trace the CMB
  cmbTh = np.linspace(0, 2*np.pi, 100)
  cmbRa = cmbRadius*np.ones(100)
  ax.plot(cmbTh, cmbRa, c='k', linestyle='-', linewidth=0.5, zorder=2)

def plotEarthSurf(ax):
  # trace the earth surface
  surfTh = np.linspace(0, 2*np.pi, 100)
  surfRa = earthRadius*np.ones(100)
  ax.plot(surfTh, surfRa, c='k', linewidth=0.5)

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


#=========================================
def doPlot(th, r, z, figID, cm, bd, fileName, label, Tvalue=-1):
  fig1 = plt.figure(figID)
  ax1 = fig1.add_subplot(111, projection='polar')

  h1=ax1.pcolormesh(th, r, z, cmap=cm, shading = "flat", vmin=bd[0],vmax=bd[1], zorder=1)

  ax1.set_ylim([cmbRadius, earthRadius])
  ax1.set_yticks([]) #[3480, 5701, 6371])
  #plt.yticks(fontsize=13)


  ax1.set_thetamin(-90)
  ax1.set_thetamax(90)
  ax1.set_xticks(np.pi/180. * np.linspace(-90, 90., 7, endpoint=True))
  ax1.set_xticklabels([r'$\pi$', r'$5\pi/6$', r'$4\pi/6$', r'$\pi/2$', r'$2\pi/6$', r'$\pi/6$', r'$0$'],fontsize=11)

  #ax1.grid(linewidth=0.1)
  ax1.set_rorigin(-1)
  plotEarthSurf(ax1)
  plotCMB(ax1)
  plotTransitionBD(ax1)
  fig1.colorbar(h1)

  if label=='fom':
    plt.text(-5, 0, 'T='+str(Tvalue), horizontalalignment='center', rotation=90,
             verticalalignment='center', fontsize=16)
    plt.text(0, 1400, 'FOM', horizontalalignment='center', rotation=90,
             verticalalignment='center', fontsize=16)
  elif label=='rom':
    plt.text(0, 1400, 'ROM', horizontalalignment='center', rotation=90,
             verticalalignment='center', fontsize=16)
  elif label=='err':
    plt.text(0, 1400, 'Error', horizontalalignment='center', rotation=90,
             verticalalignment='center', fontsize=16)

  plt.tight_layout()
  fig1.savefig('./plots/'+fileName, format="png",bbox_inches='tight', dpi=300)


###############################
if __name__== "__main__":
###############################
  # plot final velocity data for:
  # T=51.78003645 (middle of the sampling range)
  # T=69 (extrapolation)

  #load coordinates (which are the same for every case)
  nr, nth = 512, 2048
  cc = np.loadtxt("./data/coords_vp.txt")
  th, r = -cc[:,0]+np.pi/2., cc[:, 1]/1000. #m to km
  th, r = th.reshape((nr,nth)), r.reshape((nr,nth))

  Tprint = ['51.78', '69']
  Tvals  = [51.78003645, 69]
  fomFiles = ['./data/fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_0/finalFomState_vp_0',
              './data/fom_mesh512x2048_nThreads_36_dt_0.1_T_2000.0_snaps_true_seismo_true_mat_prem_fRank_1_test_11/finalFomState_vp_0']

  romFiles = ['./data/rom_mesh512x2048_nThreads_18_dt_0.1_T_2000.0_snaps_true_mat_prem_fRank_14_nPod_436_436/finalFomState_vp_0',
              './data/rom_mesh512x2048_nThreads_18_dt_0.1_T_2000.0_snaps_true_mat_prem_fRank_14_nPod_436_436/finalFomState_vp_11']

  cm1 = plt.cm.get_cmap('PuOr')
  cm2 = plt.cm.get_cmap('BrBG_r')
  cm3 = plt.cm.get_cmap('PiYG')
  for T,fom,rom in zip(Tprint[1:], fomFiles[1:], romFiles[1:]):
    print(T,fom,rom)

    # fom
    fomState = np.loadtxt(fom, skiprows=1)
    fileName = 'fom_T_'+str(T)+'.png'
    doPlot(th, r, fomState.reshape((nr, nth)), 0, cm1, [-3e-10, 3e-10], fileName, 'fom', T)

    # rom
    romState = np.loadtxt(rom, skiprows=1)
    fileName = 'rom_436_T_'+str(T)+'.png'
    doPlot(th, r, romState.reshape((nr, nth)), 1, cm1, [-3e-10, 3e-10], fileName, 'rom')

    # computeErrors(fomState, romState, "vp")
    error = fomState-romState
    fileName = 'error_T_'+str(T)+'.png'
    doPlot(th, r, error.reshape((nr, nth)), 2, cm1, [-3e-10, 3e-10], fileName, 'err')

    plt.show()
