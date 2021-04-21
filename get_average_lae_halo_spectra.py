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
import sys

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
sources = sources[np.argsort(sources["flux"])]

sources["redshift"] = sources["wave"]/1215.67 - 1
z_max = np.nanmax(sources["redshift"])
z_min = np.nanmin(sources["redshift"])
min_bin_step = 2./(1+z_max) # narrowest bins
min_bins = np.arange(3470/(1+z_max), 5542/(1+z_min) + min_bin_step, min_bin_step)

big_list = {"0_2":[], "2_5":[], "5_10":[]}
big_list_loc = {"0_2":[], "2_5":[], "5_10":[]}
N_list = []
for i,source in enumerate(sources):

    redshift = source["redshift"]
    fin = f"core_spectra/lae_{source['detectid']}.dat"

    tmp = ascii.read(fin)
    for j, appendix in enumerate(["0_2", "2_5", "5_10"]):
        tmp["spec_"+appendix] *= 0.5 * (1+redshift)**2 # factor 1/2 to go from 1/(2A) to 1/A binning, redshift factor for ergs and 1/A
        tab = np.interp(min_bins, tmp["wave_rest"], tmp["spec_"+appendix])

        big_list[appendix].append(tab)

    fin = f"core_spectra_loc/lae_{source['detectid']}.dat"

    tmp = ascii.read(fin)
    for j, appendix in enumerate(["0_2", "2_5", "5_10"]):
        tmp["spec_"+appendix] *= 0.5 * (1+redshift)**2 # factor 1/2 to go from 1/(2A) to 1/A binning, redshift factor for ergs and 1/A
        tab = np.interp(min_bins, tmp["wave_rest"], tmp["spec_"+appendix])

        big_list_loc[appendix].append(tab)


    if i%100==0:
        print(f"Done with {i}/3692.")


stack_median = {"0_2": [], "2_5": [], "5_10": []}
stack_median_loc = {"0_2": [], "2_5": [], "5_10": []}
stack_sigma = {"0_2": [], "2_5": [], "5_10": []}
stack_sigma_loc = {"0_2": [], "2_5": [], "5_10": []}
for appendix in ["0_2", "2_5", "5_10"]:
    big_list[appendix] = np.array(big_list[appendix])
    ones = np.ones(big_list[appendix].shape)
    ones[~np.isfinite(big_list[appendix])] = 0

    stack_median[appendix] = np.nanmedian(big_list[appendix], axis=0)
    stack_median_loc[appendix] = np.nanmedian(big_list_loc[appendix], axis=0)
    N = np.nansum(ones, axis=0)
    stack_sigma[appendix] = biweight_scale(big_list[appendix], axis=0, ignore_nan=True)/np.sqrt(N)
    stack_sigma_loc[appendix] = biweight_scale(big_list_loc[appendix], axis=0, ignore_nan=True)/np.sqrt(N)

wave_plot = min_bins

plt.figure(figsize=(7,7))
rmin, rmax = 0,2
plt.plot(wave_plot, stack_median[appendix], label=r"median spectrum from r={}'' to r={}'' (full-frame skysub)".format(rmin, rmax))
plt.plot(wave_plot, stack_median_loc[appendix], label=r"median spectrum from r={}'' to r={}'' (local skysub)".format(rmin, rmax))
plt.xlim(1160, 1270)
plt.ylim(-0.1, 0.3)
plt.xlabel(r"wavelength [$\AA$]")
plt.ylabel(r"flux density [$10^{-17} \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\AA^{-1}$]")
plt.savefig("med_spec_lya_tmp.png", bbox_inches="tight")

ylims = {"0_2": (-0.2, 1.75), "2_5": (-0.05, 0.15), "5_10": (-0.05, 0.15)}
#{"0_2": (-0.1, 0.15), "2_5": (-0.05, 0.15), "5_10": (-0.05, 0.15)} 
fig = plt.figure(figsize=(12,12))
for i, appendix in enumerate(["0_2", "2_5", "5_10"]):

    ax = fig.add_subplot(3,1,i+1)

    if False:
        for tab in big_list[appendix]:
            ax.plot(wave_plot, tab)

    rmin, rmax = appendix.split("_")
    ax.plot(wave_plot, stack_median[appendix], label=r"median spectrum from r={}'' to r={}'' (full-frame skysub)".format(rmin, rmax))
    ax.plot(wave_plot, stack_median_loc[appendix], label=r"median spectrum from r={}'' to r={}'' (local skysub)".format(rmin, rmax), color="tab:orange")
    
    #ax.fill_between(wave_plot, stack_median[appendix] - stack_sigma[appendix], stack_median[appendix] + stack_sigma[appendix], alpha=0.5, color="tab:blue")
    #ax.fill_between(wave_plot, stack_median_loc[appendix] - stack_sigma_loc[appendix], stack_median_loc[appendix] + stack_sigma_loc[appendix], alpha=0.5, color="tab:orange")

    ax.axvline(1215.67, linestyle=":", color="gray")
    ax.axvline(1549, linestyle=":", color="gray")
    ax.axvline(1551, linestyle=":", color="gray")
    ax.text(1552-30, 0.08, "C IV")

    ax.axvline(1640, linestyle=":", color="gray")
    ax.text(1641-30, 0.08, "He II")

    ax.axvline(1661, linestyle=":", color="gray")
    ax.axvline(1666, linestyle=":", color="gray")
    ax.text(1669, 0.08, "O III]")

    ax.set_xlim(900, 1800)
    ax.set_ylabel(r"flux density [$10^{-17} \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\AA^{-1}$]")
    ax.set_ylim(ylims[appendix][0], ylims[appendix][1])
    ax.grid(alpha=0.5)
    ax.legend(loc="upper right")
ax.set_xlabel(r"wavelength [$\AA$]")
fig.set_facecolor("white")
fig.savefig("average_lae_halo_spectra_all_compare_skysubs.pdf", bbox_inches="tight")

