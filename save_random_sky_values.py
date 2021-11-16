import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9
import astropy.units as u
import tables as tb
import sys
import os
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
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir_apx", type=str, default="_newflag",
                    help="Directory appendix.")
parser.add_argument('-s', '--final_dir', type=str, default=".", help='Directory to save radial profiles. This is necessary to add.')
parser.add_argument('--mask_continuum', type=str, default='False', help='Mask continuum fibers: True or False.')
parser.add_argument('--sn_65', type=str, default='True', help='Use only sources with S/N>6.5: True or False.')
parser.add_argument('--sample', type=int, default=3, help='Which sample to use? 1: broad lines. 2: narrow lines, high L. 3: narrow lines, low L.')
parser.add_argument('--intwidth', type=str, default='', help='Appendix for the fixed integration width: nothing, _4, or _11.')
parser.add_argument('-b', '--bootstrap', type=int, default=200, help='Number of bootstrapping samples.')
args = parser.parse_args(sys.argv[1:])

fmtstr = " Name: %(user_name)s : %(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s"
datestr = "%m/%d/%Y %I:%M:%S %p "
#basic logging config
logging.basicConfig(
	filename=os.path.join(args.final_dir, "random_sky_value.log"),
	level=logging.DEBUG,
	filemode="w",
	datefmt=datestr,
)


DIR_APX = args.dir_apx
logging.info("Directory appendix: "+ DIR_APX)

if args.final_dir is ".":
	logging.error("You must provide a directory to save the radial profiles with -s DIRECTORY.")
	sys.exit()
final_dir = args.final_dir

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
	
	u = (1 - u ** 2) ** 2
	
	weights[~np.isfinite(weights)] = 0
	
	u = u + weights**2
	u[weights==0] = 0
	d[weights==0] = 0
	u[mask] = 0

	# along the input axis if data is constant, d will be zero, thus
	# the median value will be returned along that axis
	return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)

# read in data
sources = ascii.read("../karls_suggestion/high_sn_sources_combined.tab")
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

SN_65 = (args.sn_65 == 'True')
if SN_65:
	logging.info('Including only S/N>6.5.')
	total_mask = total_mask * (sources['sn']>6.5)
else:
	logging.info('Including all S/N>5.5.')

narrow_lines = sources["linewidth_km/s"] < 1000/2.35
low_luminosity = sources["luminosity"] < 10**43

broad_line_sample = ~narrow_lines
logging.info('1. The broad line sample contains {} sources.'.format(len(broad_line_sample[broad_line_sample])))

narrow_line_high_L_sample = narrow_lines * (~low_luminosity)
logging.info('2. The narrow line, high L sample contains {} sources.'.format(len(narrow_line_high_L_sample[narrow_line_high_L_sample])))

narrow_line_low_L_sample = narrow_lines * low_luminosity
logging.info('3. The narrow line, low L sample contains {} sources.'.format(len(narrow_line_low_L_sample[narrow_line_low_L_sample])))

sample = int(args.sample)
print('args.sample: ', sample, type(sample))
SAMPLE_1 = sample == 1
SAMPLE_2 = sample == 2
SAMPLE_3 = sample == 3
if (not SAMPLE_1) * (not SAMPLE_2) * (not SAMPLE_3):
	logging.error('The --sample option must have a 1, 2, or 3. Cancelling this script.')
	sys.exit()

if SAMPLE_1:
	logging.info('Using broad line sample.')
	sources = sources[total_mask * broad_line_sample]
elif SAMPLE_2:
	logging.info('Using narrow line, high L sample.')
	sources = sources[total_mask * narrow_line_high_L_sample]
else:
	logging.info('Using narrow line, low L sample.')
	sources = sources[total_mask * narrow_line_low_L_sample]
	
logging.info("len(sources) = {}".format( len(sources)))

lae_masks = {
	"all": [True for i in range(len(sources))],
	"z<2.5" : sources["redshift"] < 2.5,
	"z>=2.5": sources["redshift"] >= 2.5,
	"sn>10": sources["sn"] > 10,
	"sn<=10": sources["sn"] <= 10,
	"L>median(L)": sources["luminosity"] > np.nanmedian(sources["luminosity"]),
	"L<=median(L)": sources["luminosity"] <= np.nanmedian(sources["luminosity"]),
	"linewidth<=median(linewidth)": sources["linewidth_km/s"] <= np.nanmedian(sources["linewidth_km/s"]),
	"linewidth>median(linewidth)": sources["linewidth_km/s"] > np.nanmedian(sources["linewidth_km/s"])
}

long_lae_masks = {}
for key in lae_masks.keys():
	long_lae_masks[key] = []

# new bogus LAEs

savedir = "/scratch/05865/maja_n"

logging.info("Reading bogus LAEs...")
bogus_list = []
i=0
N = len(sources)
for source_idx in range(len(sources)):
	source = sources[source_idx]
	detectid = source["detectid"]
	for j in range(3):
		lae_file = os.path.join(savedir,"radial_profiles/bogus_laes{}/bogus_{}_{}.dat".format(DIR_APX, j,detectid))
		try:
			lae_file = glob.glob(lae_file)[0]
			lae_tab = ascii.read(lae_file)
			lae_tab['redshift'] = [source['redshift'] for k in range(len(lae_tab))]
			lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
			lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
			bogus_list.append(lae_tab)


			for mask_name in lae_masks.keys():
				for j in range(len(lae_tab)):
					long_lae_masks[mask_name].append(lae_masks[mask_name][source_idx])


			i+=1
			if i%100 == 0:
				logging.info(f"Finished {i}/{N}")
		except Exception as e:
			logging.error(e)
			logging.error("Failed to read "+lae_file)

bogus_tab = vstack(bogus_list)
bogus_tab["mask_7"] = bogus_tab["mask_7"].astype(bool)
bogus_tab["mask_10"] = bogus_tab["mask_10"].astype(bool)
logging.info("See if masking works: {}".format( len(bogus_tab), len(bogus_tab[bogus_tab["mask_7"]])))


# bogus values
bogus_dict = {}
fiber_area = np.pi*(0.75)**2
for key in ["flux", "flux_troughsub", "flux_4", "flux_troughsub_4", "flux_11", "flux_troughsub_11"]:
	bogus_tab[key] = bogus_tab[key] / fiber_area

MASK_CONTINUUM_FIBERS = (args.mask_continuum == 'True')
logging.info("mask continuum fibers: {}".format( MASK_CONTINUUM_FIBERS))
if MASK_CONTINUUM_FIBERS:
	for mask_name in long_lae_masks.keys():
		long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[bogus_tab["mask_7"]]
	bogus_tab = bogus_tab[bogus_tab["mask_7"]]
NEW_FLAG = False
logging.info("\nmasking with Karl's new flagging method: {}".format( NEW_FLAG))
if NEW_FLAG:
	for mask_name in long_lae_masks.keys():
		long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[long_tab["new_mask_5"]]
	long_tab = long_tab[long_tab["new_mask_5"]]



EXTENSION = "flux_troughsub"
logging.info(f'Extension: {EXTENSION}')

for mask_name in long_lae_masks.keys():
	mask = long_lae_masks[mask_name]
	total_randoms = bogus_tab[EXTENSION][mask]
	total_redshifts = bogus_tab['redshift'][mask]
	N = len(total_randoms)
	bogus_dict["median_troughsub_"+mask_name] = [np.nanmedian(total_randoms)]
	bogus_dict["median_troughsub_"+mask_name+'_ra'] = [np.nanmedian(total_randoms*(1+total_redshifts)**4)]
	bogus_dict["perc_lower_median_troughsub_"+mask_name] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
	bogus_dict["perc_upper_median_troughsub_"+mask_name] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]
	bogus_dict["err_median_troughsub_"+mask_name] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]

total_randoms = bogus_tab["flux"]
N = len(total_randoms)
bogus_dict["median_no_troughsub_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_no_troughsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_no_troughsub_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_no_troughsub_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_troughsub"]
N = len(total_randoms)
bogus_dict["mean_troughsub_all"] = [np.nanmean(total_randoms)]
bogus_dict["err_mean_troughsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_mean_troughsub_perc_lower_all"] = [abs(np.nanmean(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_mean_troughsub_perc_upper_all"] = [abs(np.nanmean(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_troughsub"]
N = len(total_randoms)
bogus_dict["biweight_troughsub_all"] = [biweight_location(total_randoms, ignore_nan=True)]
bogus_dict["err_biweight_troughsub_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_biweight_troughsub_perc_lower_all"] = [abs(biweight_location(total_randoms, ignore_nan=True)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_biweight_troughsub_perc_upper_all"] = [abs(biweight_location(total_randoms, ignore_nan=True)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_troughsub_4"]
N = len(total_randoms)
bogus_dict["median_troughsub_4_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_troughsub_4_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_troughsub_4_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_troughsub_4_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]

total_randoms = bogus_tab["flux_troughsub_11"]
N = len(total_randoms)
bogus_dict["median_troughsub_11_all"] = [np.nanmedian(total_randoms)]
bogus_dict["err_median_troughsub_11_all"] = [biweight_scale(total_randoms, ignore_nan=True)/np.sqrt(N)]
bogus_dict["err_median_troughsub_11_perc_lower_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 16))/np.sqrt(N)]
bogus_dict["err_median_troughsub_11d_perc_upper_all"] = [abs(np.nanmedian(total_randoms)-np.nanpercentile(total_randoms, 84))/np.sqrt(N)]


############ bootstrapping to get the standard error of the median ############################################################################
total_randoms = bogus_tab['flux_troughsub']
total_redshifts = bogus_tab['redshift']
total_redshifts = total_redshifts[np.isfinite(total_randoms)]
total_randoms = total_randoms[np.isfinite(total_randoms)]
total_indices = np.arange(len(total_randoms))

B = args.bootstrap
N = len(total_randoms)
logging.info('Starting bootstrapping with B={} and N={}.'.format(B, N))
sample_medians = []
sample_medians_ra = []
for i in range(B):
	new_sample = random.choices(total_indices, k=N) # Return a k sized list of elements chosen from the population with replacement.
	sample_medians.append(np.nanmedian(total_randoms[new_sample]))
	sample_medians_ra.append(np.nanmedian(total_randoms[new_sample]*(1+total_redshifts[new_sample])**4))
	logging.info('Finished {}.'.format(i))
mean_of_medians = np.nanmean(sample_medians)
std_of_medians = np.nanstd(sample_medians)
std_of_medians_ra = np.nanstd(sample_medians_ra)

bogus_dict['mean_of_medians_troughsub_all'] = [mean_of_medians]
bogus_dict['std_of_medians_troughsub_all'] = [std_of_medians]
bogus_dict['std_of_medians_troughsub_all_ra'] = [std_of_medians_ra]

savefile = os.path.join(final_dir, "random_sky_values_multimasks{}.tab".format(DIR_APX))
ascii.write(bogus_dict, savefile)
logging.info("Wrote to {}".format(savefile))
