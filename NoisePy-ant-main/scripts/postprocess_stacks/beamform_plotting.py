"""
Script to do beamforming from the stacked cross-correlations. Inspired by:
Bowden, D. C., Sager, K., Fichtner, A., & Chmiel, M. (2021).
Connecting beamforming and kernel-based noise source inversion.
Geophysical Journal International, 224(3), 1607–1620. https://doi.org/10.1093/gji/ggaa539
"""

import numpy as np
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from obspy.signal.filter import bandpass
import logging
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Set fontsize for plotting
mpl.rc('font', **{'size': 20})


def plot_beam(P, title="Beamform", save=0, savename='none', cmax=0):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.6])  # x0,y0,dx,dy
    cmap = plt.get_cmap('inferno')
    i = plt.pcolor(ux - dux / 2, uy - duy / 2, np.real(P.T), cmap=cmap, rasterized=True)  # ,vmin=-4,vmax=4)
    if cmax == 0:
        cmax = np.max(np.abs(P))
    cmin = np.min(P)
    plt.clim(cmin, cmax)

    plt.axis('equal')
    plt.axis('tight')
    plt.xlim(min(ux) + dux, max(ux) - dux)
    plt.ylim(min(uy) + duy, max(uy) - duy)
    plt.xlabel('Slowness East-West [s/km]')
    plt.ylabel('Slowness North-South [s/km]')
    ax.tick_params(top=True, right=True)
    plt.plot([np.min(ux), np.max(ux)], [0, 0], 'w')
    plt.plot([0, 0], [np.min(uy), np.max(uy)], 'w')
    theta = np.linspace(0, 2 * np.pi, 150)
    for radius in [0.1, 0.2, 0.3, 0.4, 0.5]:
        plt.plot(radius * np.cos(theta), radius * np.sin(theta), "w--")
        plt.text(radius, 0, f"{radius}", c="w")
    plt.text(0, 0.83 * max(uy), "N", c="w", fontsize=30)
    plt.text(0.83 * max(ux), 0, "E", c="w", fontsize=30)
    plt.text(0, 0.9 * min(uy), "S", c="w", fontsize=30)
    plt.text(0.9 * min(ux), 0, "W", c="w", fontsize=30)
    plt.title(title)
    colorbar_ax = fig.add_axes([0.75, 0.1, 0.03, 0.6])  # x0,y0,dx,dy
    cbar = fig.colorbar(i, cax=colorbar_ax)
    i.set_clim(0,30)
    
    if (save == 1):
        plt.savefig(savename, bbox_inches='tight', format="PNG")
    plt.close()


ncf_file = "/srv/beegfs/scratch/shares/cdff/DPM/Postprocessing/DPM_ncfs_Allstack_pws_ZZ.npz" #sys.argv[1]  # "/home/users/s/savardg/extract_ncfs/aargau/aargau_ncfs_wCH_pure_Allstack_linear_ZZ.npz"
beam_file = "/srv/beegfs/scratch/shares/cdff/DPM/Postprocessing/DPM_ncfs_Allstack_pws_ZZ_beam.npz" #ncf_file.replace(".npz", "_beam.npz")
fig_file = "/srv/beegfs/scratch/shares/cdff/DPM/Postprocessing/DPM_ncfs_Allstack_pws_ZZ_beam_norm.png" #beam_file.replace(".npz", ".png")
#logging.info(f"Input file: {ncf_file}")
logging.info(f"Output file: {beam_file}")
logging.info(f"Figure file: {fig_file}")

# Load input data
data = np.load(beam_file)
Paz = data["Paz"]  # Azimuth of each station pair
ux = data["ux"]  # Backazimuth of each station pair
uy = data["uy"]  # Inter-station distances
dux = np.abs(ux[1] - ux[0])  # Slowness spacing in x
duy = np.abs(uy[1] - uy[0])  # slowness spacing in y

# For plotting:
P = Paz #+ Pbaz # Decide if plotting beam constructed with the CCFs' positive lags or negative lag or their sum
plot_beam(P, title=os.path.split(ncf_file)[1], save=True, savename=fig_file)
