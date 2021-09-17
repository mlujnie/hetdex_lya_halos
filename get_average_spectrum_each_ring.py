import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.stats import biweight_scale
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)
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

# read in data
sources = ascii.read("../karls_suggestion/high_sn_sources.tab")
sources = sources[sources["mask"]==1]

# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
sources["redshift"] = z_lae
sources["luminosity"] = (sources["flux_213"])*1e-17*4*np.pi*(cosmo.luminosity_distance(sources["redshift"]).to(u.cm)/u.cm)**2
c = 3*10**5 # km/s
doppler_v_of = c * sources["linewidth"] / sources["wave"]
sources["linewidth_km/s"] = doppler_v_of

total_mask = np.ones(len(sources), dtype=bool)

SN_65 = True #(args.sn_65 == 'True')
if SN_65:
	print('Including only S/N>6.5.')
	total_mask = total_mask * (sources['sn']>6.5)
else:
	print('Including all S/N>5.5.')

narrow_lines = sources["linewidth_km/s"] < 1000/2.35
low_luminosity = sources["luminosity"] < 10**43
low_continuum = sources["gmag"] > 24
sfg_sample = narrow_lines & low_luminosity & low_continuum
agn_sample = ~sfg_sample

SFG_SAMPLE = True #(args.sfg == 'True')
AGN_SAMPLE = ~SFG_SAMPLE

if SFG_SAMPLE:
	print('Using SFG sample.')
	sources = sources[total_mask * sfg_sample]
else:
	print('Using AGN-dominated sample.')
	sources = sources[total_mask * agn_sample]

print("len(sources) = {}".format( len(sources)))


z_max = np.nanmax(sources["redshift"])
z_min = np.nanmin(sources["redshift"])
min_bin_step = 2./(1+z_max) # narrowest bins
min_bins = np.arange(3470/(1+z_max), 5542/(1+z_min) + min_bin_step, min_bin_step)

big_list = {}
r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320, 800])
for r_min, r_max in zip(r_bins_kpc, r_bins_max_kpc):
	appendix = f"{int(r_min)}_{int(r_max)}"
	big_list[appendix] = []
	for i,source in enumerate(sources):

		redshift = source["redshift"]
		fin = f"/scratch/05865/maja_n/core_spectra_unflagged_newflag_100/{appendix}/lae_{source['detectid']}.dat"

		tmp = ascii.read(fin)
		tab = np.interp(min_bins, tmp["wave_rest"], tmp["spec_troughsub_uf"])

		big_list[appendix].append(tab)

		if i%100==0:
			print(f"Done with {i}/3692.")


stack_median = {}
stack_sigma = {}
for appendix in big_list.keys():
	big_list[appendix] = np.array(big_list[appendix])
	ones = np.ones(big_list[appendix].shape)
	ones[~np.isfinite(big_list[appendix])] = 0

	stack_median["s_"+appendix] = np.nanmedian(big_list[appendix], axis=0)
	N = np.nansum(ones, axis=0)
	stack_sigma[appendix] = biweight_scale(big_list[appendix], axis=0, ignore_nan=True)/np.sqrt(N)
	
	stack_median["e_"+appendix] = stack_sigma[appendix]

stack_median["wavelength"] = min_bins
ascii.write(stack_median, "/scratch/05865/maja_n/core_spectra_unflagged_newflag_100/median_spectrum_all_rings.tab")
