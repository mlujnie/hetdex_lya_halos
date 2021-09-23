import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
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
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir_apx", type=str, default="_newflag",
                    help="Directory appendix.")
parser.add_argument('-s', '--final_dir', type=str, default=".", help='Directory to save radial profiles. This is necessary to add.')
parser.add_argument('--mask_continuum', type=str, default='False', help='Mask continuum fibers: True or False.')
parser.add_argument('--sn_65', type=str, default='True', help='Use only sources with S/N>6.5: True or False.')
parser.add_argument('--sample', type=int, default=3, help='Which sample to use? 1: broad lines. 2: narrow lines, high L. 3: narrow lines, low L.')
parser.add_argument('--intwidth', type=str, default='', help='Appendix for the fixed integration width: nothing, _4, or _11.')
parser.add_argument("-f", "--farout", type=str, default="False", help="Measure surface brightness out to 100''.")
args = parser.parse_args(sys.argv[1:])

fmtstr = " Name: %(user_name)s : %(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s"
datestr = "%m/%d/%Y %I:%M:%S %p "
#basic logging config
logging.basicConfig(
        filename=os.path.join(args.final_dir, "save_lutz_errors.log"),
        level=logging.DEBUG,
        filemode="w",
        datefmt=datestr,
)


DIR_APX = args.dir_apx
if args.farout == "True":
	FAR_OUT = True
	#DIR_APX = DIR_APX + "_100"

	r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
	r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320, 800])
	r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
	r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

	print("Going out to 100''.")
elif args.farout == "False":
	FAR_OUT = False

	r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160])
	r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
	r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
	r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

	print("Going out to 20''.")
else:
	print("Could not identify a valid argument for farout: True or False.")

logging.info("Directory appendix: "+ DIR_APX)

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

save_dir = "/scratch/05865/maja_n"
basedir = "/work2/05865/maja_n/stampede2/master"

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
	"linewidth>median(linewidth)": sources["linewidth_km/s"] > np.nanmedian(sources["linewidth_km/s"])}

# getting the stacks
kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

#r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160])
#r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
#r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
#r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

r_bins = r_bins_kpc / kpc_per_arcsec_mid
r_bins_max = r_bins_max_kpc / kpc_per_arcsec_mid
r_bins_plot = r_bins_plot_kpc / kpc_per_arcsec_mid 
r_bins_xbars = (r_bins_max-r_bins)/2.


# shifted LAE data
all_shifts = {}
for d_wl in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
			-10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100, -105]:

	long_lae_masks = {}
	for key in lae_masks.keys():
		long_lae_masks[key] = []


	save_file = os.path.join(args.final_dir, "radial_profiles_empty_wlshift/radial_profiles_proper_multimask_{}_{}.tab".format(DIR_APX, d_wl))
	if False: #os.path.exists(save_file):
		logging.info(f"Skipping {save_file}: already exists.")
		continue

	logging.info("Reading LAEs... {}".format( d_wl))
	long_list = []
	i=0
	N = len(sources)
	for source_idx in range(len(sources)):
		source = sources[source_idx]
		detectid = source["detectid"]
		wavelength = source["wave"]
		redshift = wavelength/1215.67 - 1
		lae_file = os.path.join(save_dir, "radial_profiles/laes_wloffset{}/lae_{}_{}.dat".format(DIR_APX,detectid, d_wl))
		lae_file = glob.glob(lae_file)[0]
		try:
			lae_tab = ascii.read(lae_file)
			lae_tab["detectid"] = [detectid for j in range(len(lae_tab))]
			lae_tab["redshift"] = [redshift for j in range(len(lae_tab))]
			lae_tab["mask_7"] = lae_tab["mask_7"].astype(str)=="True"
			lae_tab["mask_10"] = lae_tab["mask_10"].astype(str)=="True"
			long_list.append(lae_tab)

			for mask_name in lae_masks.keys():
				for j in range(len(lae_tab)):
					long_lae_masks[mask_name].append(lae_masks[mask_name][source_idx])

			i+=1
			if i%100 == 0:
				logging.info(f"Finished {i}/{N}")
		except Exception as e:
			logging.error("Failed to read "+lae_file)
	try: 
		long_tab = vstack(long_list)
	except Exception as e:
		logging.error("Vstack failed:")
		logging.error(e)
	#long_tab["mask_7"] = (long_tab["mask_7"]=="True").astype(bool)
	#long_tab["mask_10"] = (long_tab["mask_10"]=="True").astype(bool)

	MASK_CONTINUUM_FIBERS = (args.mask_continuum == 'True')
	logging.info("mask continuum fibers: {}".format( MASK_CONTINUUM_FIBERS))
	if MASK_CONTINUUM_FIBERS:
		for mask_name in long_lae_masks.keys():
			long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[long_tab["mask_7"]]
		long_tab = long_tab[long_tab["mask_7"]]

	NEW_FLAG = False
	logging.info("\nmasking with Karl's new flagging method: {}".format( NEW_FLAG))
	if NEW_FLAG:
		for mask_name in long_lae_masks.keys():
			long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[long_tab["new_mask_5"]]
		long_tab = long_tab[long_tab["new_mask_5"]]


	# getting the stacks
	kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

	#r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160])
	#r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
	#r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
	#r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

	r_bins = r_bins_kpc / kpc_per_arcsec_mid
	r_bins_max = r_bins_max_kpc / kpc_per_arcsec_mid
	r_bins_plot = r_bins_plot_kpc / kpc_per_arcsec_mid 
	r_bins_xbars = (r_bins_max-r_bins)/2.

	big_tab_proper = {}

	EXTENSION = "flux_troughsub"+args.intwidth
	logging.info(f"Extension: {EXTENSION}")


	for mask_name in lae_masks.keys():
		here = long_lae_masks[mask_name]
		r_kpc, sb_median_kpc, e_sb_median_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab[EXTENSION][here], long_tab["redshift"][here])
		big_tab_proper["median_troughsub_"+mask_name] = sb_median_kpc
		big_tab_proper["err_median_troughsub_"+mask_name] = e_sb_median_kpc

	big_tab_proper["r/kpc"] = r_kpc
	big_tab_proper["delta_r/kpc"] = r_bins_kpc_xbars

	r_kpc, big_tab_proper["median_no_troughsub_all"], big_tab_proper["err_median_no_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab["flux"], long_tab["redshift"])
	r_kpc, big_tab_proper["mean_troughsub_all"], big_tab_proper["err_mean_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="mean")
	r_kpc, big_tab_proper["biweight_troughsub_all"], big_tab_proper["err_biweight_contsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="biweight")

	save_file = os.path.join(args.final_dir, "radial_profiles_empty_wlshift/radial_profiles_proper_multimask{}_{}.tab".format(DIR_APX, d_wl))
	ascii.write(big_tab_proper, save_file)
	logging.info("Wrote to {}".format( save_file))
