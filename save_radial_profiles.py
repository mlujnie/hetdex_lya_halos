import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9
import astropy.units as u
import tables as tb
import sys
from scipy.interpolate import interpn
from astropy.convolution import convolve_fft
import time

import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation, biweight_scale

# read in interpolation grid
from scipy.interpolate import RegularGridInterpolator
INTERPOLATOR = RegularGridInterpolator

from scipy.ndimage import gaussian_filter
import glob
import pickle

# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)

# some functions
import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation


def biweight_location_weights(data, weights, c=6.0, M=None, axis=None):
	""" weighted biweight location a la Karl
		nan-resistant and excludes data with zero weights"""

	data = np.asanyarray(data).astype(np.float64)
	weights = np.asanyarray(weights).astype(np.float64)
   
	data[weights==0] = np.nan 
	weights[~np.isfinite(data)] = np.nan
	
	if (data.shape!=weights.shape):
		raise ValueError("data.shape != weights.shape")

	if M is None:
		M = np.nanmedian(data, axis=axis)
	if axis is not None:
		M = np.expand_dims(M, axis=axis)

	# set up the differences
	d = data - M

	# set up the weighting
	mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
	#madweights = median_absolute_deviation(weights, axis=axis)

	if axis is None and mad == 0.:
		return M  # return median if data is a constant array
	
	#if axis is None and madweights == 0:
	#	madweights = 1.

	if axis is not None:
		mad = np.expand_dims(mad, axis=axis)
		const_mask = (mad == 0.)
		mad[const_mask] = 1.  # prevent divide by zero

	#if axis is not None:
	#	madweights = np.expand_dims(madweights, axis=axis)
	#	const_mask = (madweights == 0.)
	#	madweights[const_mask] = 1.  # prevent divide by zero

	cmadsq = (c*mad)**2
	
	factor = 0.5
	weights  = weights/np.nanmedian(weights)*factor 
	
	u = d / (c * mad)

	# now remove the outlier points
	mask = (np.abs(u) >= 1)
	#print("number of excluded points ", len(mask[mask]))
	
	u = (1 - u ** 2) ** 2
	
	weights[~np.isfinite(weights)] = 0
	
	u = u + weights**2
	u[weights==0] = 0
	d[weights==0] = 0
	u[mask] = 0

	# along the input axis if data is constant, d will be zero, thus
	# the median value will be returned along that axis
	return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)

def lae_powerlaw_profile(r, c1, c2):
	return c1*psf_func((FWHM, r)) + c2*powerlaw_func((FWHM, r))

def fit_moffat(dist, amp, fwhm):
	beta = 3.
	gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
	norm = (beta-1)/(np.pi*gamma**2)
	return amp * norm * (1+(dist/gamma)**2)**(-1*beta)

def get_stack(r_bins_min, r_bins_max, data_r, data_flux, kind="median", weights_flux=None):
	
	stack, e_stack = [], []
	for r_min, r_max in zip(r_bins_min, r_bins_max):
		here = (data_r < r_max)&(data_r >= r_min)
		tmp_flux = data_flux[here]
		if weights_flux is not None:
			tmp_weights = weights_flux[here]
		if kind == "median":
			stack.append(np.nanmedian(tmp_flux))
			e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
		elif kind == "mean":
			stack.append(np.nanmean(tmp_flux))
			e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
		elif kind == "biweight":
			stack.append(biweight_location(tmp_flux, ignore_nan=True))
			e_stack.append(biweight_scale(tmp_flux, ignore_nan=True)/np.sqrt(len(tmp_flux)))
		elif kind == "weighted_biweight":
			if weights_flux is None:
				print("You need to provide weights!")
				return 0, 0, 0
			mask = np.isfinite(tmp_flux)&(np.isfinite(tmp_weights))
			stack.append(biweight_location_weights(tmp_flux[mask], tmp_weights[mask]))
			e_stack.append(biweight_scale(tmp_flux[mask])/np.sqrt(len(tmp_flux[mask])))
		else:
			print("Unknown kind: ", kind)
			return 0, 0, 0
	r_bins_mid = np.nanmean([r_bins_min, r_bins_max], axis=0)
	fiberarea = np.pi*0.75**2
	return r_bins_mid, np.array(stack)/fiberarea, np.array(e_stack)/fiberarea

def get_stack_proper(r_bins_min, r_bins_max, data_r, data_flux, data_redshift, kind="median", weights_flux=None):
	
	stack, e_stack = [], []
	kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(data_redshift)/60*u.arcmin/u.kpc
	for r_min, r_max in zip(r_bins_min, r_bins_max):
		here = (data_r*kpc_per_arcsec < r_max)&(data_r*kpc_per_arcsec >= r_min)
		tmp_flux = data_flux[here]
		if weights_flux is not None:
			tmp_weights = weights_flux[here]
		if kind == "median":
			stack.append(np.nanmedian(tmp_flux))
			e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
		elif kind == "mean":
			stack.append(np.nanmean(tmp_flux))
			e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
		elif kind == "biweight":
			stack.append(biweight_location(tmp_flux, ignore_nan=True))
			e_stack.append(biweight_scale(tmp_flux, ignore_nan=True)/np.sqrt(len(tmp_flux)))
		elif kind == "weighted_biweight":
			if weights_flux is None:
				print("You need to provide weights!")
				return 0, 0, 0
			mask = np.isfinite(tmp_flux)&(np.isfinite(tmp_weights))
			stack.append(biweight_location_weights(tmp_flux[mask], tmp_weights[mask]))
			e_stack.append(biweight_scale(tmp_flux[mask])/np.sqrt(len(tmp_flux[mask])))
		else:
			print("Unknown kind: ", kind)
			return 0, 0, 0
	r_bins_mid = np.nanmean([r_bins_min, r_bins_max], axis=0)
	fiberarea = np.pi*0.75**2
	return r_bins_mid, np.array(stack)/fiberarea, np.array(e_stack)/fiberarea

# read in data
sources = ascii.read("../karls_suggestion/high_sn_sources.tab")
sources["mask"] = sources["mask"].astype(bool)
sources = sources[sources["mask"]]
sources = sources[sources["sn"]>6.5]

# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
sources["redshift"] = z_lae
luminositites = sources["flux_213"]*cosmo.luminosity_distance(z_lae)**2*4*np.pi/u.Mpc**2
sources["luminosity"] = luminositites
c = 3*10**8 # m/s
sources["linewidth_km/s"] = c*sources["linewidth"]/sources["wave"] *1/1000.

lae_masks = {
	"all": [True for i in range(len(sources))],
	"sn>6.5" : sources["sn"] > 6.5,
	"sn<=6.5" : sources["sn"] <= 6.5,
	"linewidth<1000km/s": sources["linewidth_km/s"] < 1000,
	"linewidth>=1000km/s": sources["linewidth_km/s"] >= 1000,
	"z<2.75" : sources["redshift"] < 2.75,
	"z>=2.75": sources["redshift"] >= 2.75,
	"sn>10": sources["sn"] > 10,
	"sn<=10": sources["sn"] <= 10,
	"L>mean(L)": sources["luminosity"] > np.nanmean(sources["luminosity"]),
	"L<=mean(L)": sources["luminosity"] <= np.nanmean(sources["luminosity"]),
	"linewidth<=mean(linewidth)": sources["linewidth_km/s"] <= np.nanmean(sources["linewidth_km/s"]),
	"linewidth>mean(linewidth)": sources["linewidth_km/s"] > np.nanmean(sources["linewidth_km/s"])
}

long_lae_masks = {}
for key in lae_masks.keys():
	long_lae_masks[key] = []

# real LAE data
print("Reading LAEs...")
long_list = []
i=0
N = len(sources)
for source_idx in range(len(sources)):
	source = sources[source_idx]

	detectid = source["detectid"]
	lae_file = "../radial_profiles/laes_skymask/lae_{}.dat".format(detectid)
	lae_file = glob.glob(lae_file)[0]
	try:
		lae_tab = ascii.read(lae_file)
		if len(lae_tab) < 1:
			continue
		lae_tab["detectid"] = [detectid for j in range(len(lae_tab))]
		lae_tab["redshift"] = [source["redshift"] for j in range(len(lae_tab))]
		lae_tab["mask_7"] = lae_tab["mask_7"].astype(str)
		long_list.append(lae_tab)

		for mask_name in lae_masks.keys():
			for j in range(len(lae_tab)):
				long_lae_masks[mask_name].append(lae_masks[mask_name][source_idx])

		i+=1
		if i%100 == 0:
			print(f"Finished {i}/{N}")
	except Exception as e:
		print("Failed to read "+lae_file)
print("{} LAEs are non-empty.".format(i))
		
long_tab = vstack(long_list)
long_tab["mask_7"] = (long_tab["mask_7"]=="True").astype(bool)
long_tab["mask_10"] = (long_tab["mask_10"]=="True").astype(bool)

print("See if masking works: ", len(long_tab), len(long_tab[long_tab["mask_7"]]))
for mask_name in long_lae_masks.keys():
	long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[long_tab["mask_7"]]
long_tab = long_tab[long_tab["mask_7"]]

# getting the stacks
kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160])
r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

r_bins = r_bins_kpc / kpc_per_arcsec_mid
r_bins_max = r_bins_max_kpc / kpc_per_arcsec_mid
r_bins_plot = r_bins_plot_kpc / kpc_per_arcsec_mid 
r_bins_xbars = (r_bins_max-r_bins)/2.

big_tab_proper = {}

EXTENSION = "flux_contsub_11"


for mask_name in lae_masks.keys():
	here = long_lae_masks[mask_name]
	r_kpc, sb_median_kpc, e_sb_median_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab[EXTENSION][here], long_tab["redshift"][here])
	big_tab_proper["median_contsub_"+mask_name] = sb_median_kpc
	big_tab_proper["err_median_contsub_"+mask_name] = e_sb_median_kpc

big_tab_proper["r/kpc"] = r_kpc
big_tab_proper["delta_r/kpc"] = r_bins_kpc_xbars

r_kpc, big_tab_proper["median_no_contsub_all"], big_tab_proper["err_median_no_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab["flux"], long_tab["redshift"])
r_kpc, big_tab_proper["mean_contsub_all"], big_tab_proper["err_mean_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="mean")
r_kpc, big_tab_proper["biweight_contsub_all"], big_tab_proper["err_biweight_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="biweight")


ascii.write(big_tab_proper, "radial_profiles_proper_multimasks_sn_65_11.tab")

