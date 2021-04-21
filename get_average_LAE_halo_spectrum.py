# wave_rest spec_0_2 spec_2_5 spec_5_10

import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.stats import biweight_scale
import glob
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# update matplotlib params for bigger font
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral'}
pylab.rcParams.update(params)


big_list = []
ff = glob.glob("core_spectra/*.dat")
for fin in ff:
    big_list.append(ascii.read(fin))

big_tab = vstack(big_list)

wave_step = 2
wavelength_bins = np.arange(np.nanmin(big_tab["wave_rest"]), np.nanmax(big_tab["wave_rest"])+wave_step, wave_step)
stack_median = {"0_2": [], "2_5": [], "5_10": []}
stack_sigma = {"0_2": [], "2_5": [], "5_10": []}
for wl_min, wl_max in zip(wavelength_bins[:-1], wavelength_bins[1:]):
    wl_here = (big_tab["wave_rest"]>= wl_min) & (big_tab["wave_rest"] < wl_max)
    for appendix in ["0_2", "2_5", "5_10"]:
        spec_tab = big_tab["spec_"+appendix][wl_here]
        stack_median[appendix].append(np.nanmedian(spec_tab))
        N = np.size(spec_tab[np.isfinite(spec_tab)])
        stack_sigma[appendix].append(biweight_scale(spec_tab, axis=0, ignore_nan=True)/np.sqrt(N))

wave_plot = wavelength_bins[:-1] + wave_step/2.

ylims = {"0_2": (-0.025, 0.075), "2_5": (-0.01, 0.01), "5_10": (-0.01, 0.01)}
fig = plt.figure(figsize=(15,15))
for i, appendix in enumerate(["0_2", "2_5", "5_10"]):

    stack_median[appendix] = np.array(stack_median[appendix])
    stack_sigma[appendix] = np.array(stack_sigma[appendix])

    ax = fig.add_subplot(3,1,i+1)
    ax.plot(wave_plot, stack_median[appendix])
    ax.fill_between(wave_plot, stack_median[appendix] - stack_sigma[appendix], stack_median[appendix] + stack_sigma[appendix], alpha=0.3)
    ax.set_xlim(900, 1800)
    rmin, rmax = appendix.split("_")
    ax.set_title("median spectrum from {}'' to {}''".format(rmin, rmax))
    ax.set_ylabel(r"flux [$10^{-17} \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\AA^{-1}$]")
    ax.set_ylim(ylims[appendix][0], ylims[appendix][1])
    ax.grid(alpha=0.5)

ax.set_xlabel(r"wavelength [$\AA$]")
fig.set_facecolor("white")
fig.savefig("average_lae_halo_spectra.pdf", bbox_inches="tight")
