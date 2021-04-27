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

# read source list
sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
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

# new bogus LAEs
print("Reading bogus LAEs...")
bogus_list = []
i=0
N = len(sources)
for source_idx in range(len(sources)):
	source = sources[source_idx]
	detectid = source["detectid"]
	lae_file = "../radial_profiles/bogus_laes_skymask/bogus_0_{}.dat".format(detectid)
	try:
		lae_file = glob.glob(lae_file)[0]
		lae_tab = ascii.read(lae_file)
		lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
		lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
		bogus_list.append(lae_tab)


		for mask_name in lae_masks.keys():
			for j in range(len(lae_tab)):
				long_lae_masks[mask_name].append(lae_masks[mask_name][source_idx])


		i+=1
		if i%100 == 0:
			print(f"Finished {i}/{N}")
	except Exception as e:
		print(e)
		break
		print("Failed to read "+lae_file)

	lae_file = "../radial_profiles/bogus_laes_skymask/bogus_1_{}.dat".format(detectid)
	try:
		lae_file = glob.glob(lae_file)[0]
		lae_tab = ascii.read(lae_file)
		lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
		lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
		bogus_list.append(lae_tab)
		i+=1
		if i%100 == 0:
			print(f"Finished {i}/{N}")


		for mask_name in lae_masks.keys():
			for j in range(len(lae_tab)):
				long_lae_masks[mask_name].append(lae_masks[mask_name][source_idx])


	except Exception as e:
		print("Failed to read "+lae_file)


bogus_tab = vstack(bogus_list)
bogus_tab["mask_7"] = bogus_tab["mask_7"].astype(bool)
bogus_tab["mask_10"] = bogus_tab["mask_10"].astype(bool)
print("See if masking works: ", len(bogus_tab), len(bogus_tab[bogus_tab["mask_7"]]))

for mask_name in long_lae_masks.keys():
	long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[bogus_tab["mask_7"]]
bogus_tab = bogus_tab[bogus_tab["mask_7"]]


# bogus values
bogus_dict = {}
fiber_area = np.pi*(0.75)**2

for key in ["flux", "flux_contsub", "flux_4", "flux_contsub_4", "flux_11", "flux_contsub_11"]:
	bogus_tab[key] = bogus_tab[key] / fiber_area

EXTENSION = "flux_contsub_11"

for mask_name in long_lae_masks.keys():
	mask = long_lae_masks[mask_name]
	total_randoms = bogus_tab[EXTENSION][mask]
	N = len(total_randoms)
	bogus_dict["median_contsub_"+mask_name] = [np.nanmedian(total_randoms)]
	bogus_dict["err_median_contsub_perc_upper_"+mask_name] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
	bogus_dict["err_median_contsub_perc_upper_"+mask_name] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]
	bogus_dict["err_median_contsub_"+mask_name] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]

total_randoms = bogus_tab["flux"]
N = len(total_randoms)
bogus_dict["median_no_contsub_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_no_contsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_no_contsub_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_no_contsub_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_contsub"]
N = len(total_randoms)
bogus_dict["mean_contsub_all"] = [np.nanmean(total_randoms)]
bogus_dict["err_mean_contsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_mean_contsub_perc_lower_all"] = [abs(np.nanmean(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_mean_contsub_perc_upper_all"] = [abs(np.nanmean(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_contsub"]
N = len(total_randoms)
bogus_dict["biweight_contsub_all"] = [biweight_location(total_randoms, ignore_nan=True)]
bogus_dict["err_biweight_contsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_biweight_contsub_perc_lower_all"] = [abs(biweight_location(total_randoms, ignore_nan=True)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_biweight_contsub_perc_upper_all"] = [abs(biweight_location(total_randoms, ignore_nan=True)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_contsub_4"]
N = len(total_randoms)
bogus_dict["median_contsub_4_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_contsub_4_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_contsub_4_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_contsub_4_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_contsub_11"]
N = len(total_randoms)
bogus_dict["median_contsub_11_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_contsub_11_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_contsub_11_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_contsub_11d_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

ascii.write(bogus_dict, "random_sky_values_multimasks_sn_65_11.tab")
print("Wrote to random_sky_values_multimasks.tab")
