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

import logging

# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)

# some functions
import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation
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
parser.add_argument("--bootstrap", type=int, default=1000, help='Number of boot strap loops for the error estimate.')
args = parser.parse_args(sys.argv[1:])

fmtstr = " Name: %(user_name)s : %(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s"
datestr = "%m/%d/%Y %I:%M:%S %p "
#basic logging config
logging.basicConfig(
	filename=os.path.join(args.final_dir, "radial_profile.log"),
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

if args.final_dir is ".":
	logging.error("You must provide a directory to save the radial profiles with -s DIRECTORY.")
	sys.exit()
final_dir = args.final_dir

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
sources = ascii.read("../karls_suggestion/high_sn_sources_combined.tab")
sources = sources[sources["mask"]==1]

# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
sources["redshift"] = z_lae
sources["luminosity"] = (sources["flux_213"])*1e-17*4*np.pi*(cosmo.luminosity_distance(sources["redshift"]).to(u.cm)/u.cm)**2
c = 3*10**5 # km/s
doppler_v_of = c * sources['linewidth'] / sources["wave"]
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

savedir = '/work2/05865/maja_n/stampede2/master/karls_suggestion' #"/scratch/05865/maja_n"

# real LAE data
logging.info("Reading LAEs...")
long_list = []
i=0
N = len(sources)
used_detectids = []
for source_idx in range(len(sources)):
	source = sources[source_idx]

	detectid = source["detectid"]
	lae_file = os.path.join(savedir, "radial_profiles/laes{}/lae_{}.dat".format(DIR_APX, detectid))
	if i == 0:
		logging.info(lae_file)
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
		used_detectids.append(detectid)
		if i%100 == 0:
			logging.info(f"Finished {i}/{N}")
	except Exception as e:
		logging.info("Failed to read "+lae_file)
logging.info("{} LAEs are non-empty.".format(i))
np.savetxt('used_detectids.txt', used_detectids)
sys.exit()
		
long_tab = vstack(long_list)
long_tab["mask_7"] = (long_tab["mask_7"]=="True").astype(bool)
long_tab["mask_10"] = (long_tab["mask_10"]=="True").astype(bool)

logging.info("See if masking works: {} {}".format( len(long_tab), len(long_tab[long_tab["mask_7"]])))

MASK_CONTINUUM_FIBERS = (args.mask_continuum == 'True')
logging.info("\nmasking continuum fibers: {}".format( MASK_CONTINUUM_FIBERS))
if MASK_CONTINUUM_FIBERS:
	for mask_name in long_lae_masks.keys():
		long_lae_masks[mask_name] = np.array(long_lae_masks[mask_name])[long_tab["mask_7"]]
	long_tab = long_tab[long_tab["mask_7"]]

NEW_FLAG = False
logging.info("\nmasking with Karl's new flagging method: {}".format(NEW_FLAG))
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

EXTENSION = "flux_troughsub" + args.intwidth
logging.info(f'Extension: {EXTENSION}')


for mask_name in lae_masks.keys():
	here = long_lae_masks[mask_name]
	r_kpc, sb_median_kpc, e_sb_median_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab[EXTENSION][here], long_tab["redshift"][here])
	big_tab_proper["median_troughsub_"+mask_name] = sb_median_kpc
	big_tab_proper["err_median_troughsub_"+mask_name] = e_sb_median_kpc

	# adjust by redshift: SB x (1+z)**4
	r_kpc, sb_median_kpc, e_sb_median_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab[EXTENSION][here]*(1+long_tab['redshift'][here])**4, long_tab["redshift"][here])
	big_tab_proper["median_troughsub_"+mask_name+'_ra'] = sb_median_kpc
	big_tab_proper["err_median_troughsub_"+mask_name+'_ra'] = e_sb_median_kpc


big_tab_proper["r/kpc"] = r_kpc
big_tab_proper["delta_r/kpc"] = r_bins_kpc_xbars

r_kpc, big_tab_proper["median_no_troughsub_all"], big_tab_proper["err_median_no_troughsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab["flux"], long_tab["redshift"])
r_kpc, big_tab_proper["mean_troughsub_all"], big_tab_proper["err_mean_troughsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="mean")
r_kpc, big_tab_proper["biweight_troughsub_all"], big_tab_proper["err_biweight_troughsub_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab[EXTENSION], long_tab["redshift"], kind="biweight")

r_kpc, big_tab_proper["median_troughsub_2_all"], big_tab_proper["err_median_troughsub_2_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab["flux_troughsub_2"], long_tab["redshift"])
r_kpc, big_tab_proper["median_redcontinuum_all"], big_tab_proper["err_median_redcontinuum_all"] = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"], long_tab["red_cont_flux"], long_tab["redshift"])


############ bootstrapping to get the standard error of the median ############################################################################
B = args.bootstrap
logging.info('Starting bootstrapping with B={}.'.format(B))

kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(long_tab['redshift'])/60*u.arcmin/u.kpc
stack  = {EXTENSION:[], 'flux_troughsub_2':[], 'red_cont_flux':[]}
stack_ra = {EXTENSION:[], 'flux_troughsub_2':[], 'red_cont_flux':[]}
fiberarea = np.pi*0.75**2
data_r = long_tab['r']
data_flux = {EXTENSION: long_tab[EXTENSION], 
            'flux_troughsub_2': long_tab['flux_troughsub_2'],
            'red_cont_flux': long_tab['red_cont_flux']}
data_redshift = long_tab['redshift']

big_tab_proper['N_fibers'] = [len(long_tab[(data_r*kpc_per_arcsec < r_max)&(data_r*kpc_per_arcsec >= r_min)]) for r_min, r_max in zip(r_bins_kpc, r_bins_max_kpc)]
big_tab_proper['N_laes'] = [len(np.unique(long_tab['detectid'][(data_r*kpc_per_arcsec < r_max)&(data_r*kpc_per_arcsec >= r_min)])) for r_min, r_max in zip(r_bins_kpc, r_bins_max_kpc)]
print(big_tab_proper.keys())

for r_min, r_max in zip(r_bins_kpc, r_bins_max_kpc):
    here = (data_r*kpc_per_arcsec < r_max)&(data_r*kpc_per_arcsec >= r_min)
    for key in stack.keys():
        tmp_flux = data_flux[key][here]
        tmp_z = data_redshift[here]
        stack[key].append(tmp_flux / fiberarea)
        stack_ra[key].append(tmp_flux * (1+tmp_z)**4 / fiberarea)

sample_medians, sample_medians_ra, Ns = {}, {}, {}
for key in stack.keys():
    stack[key] = [np.array(x) for x in stack[key]]
    stack[key] = [x[np.isfinite(x)] for x in stack[key]]
    stack_ra[key] = [np.array(x) for x in stack_ra[key]]
    stack_ra[key] = [x[np.isfinite(x)] for x in stack_ra[key]]
    M = len(stack[key])
    sample_medians[key] = [[] for i in range(M)]
    sample_medians_ra[key] = [[] for i in range(M)]
    Ns[key] = [len(stack[key][i]) for i in range(M)]
    
for i in range(B):
    for j in range(M):
        for key in stack.keys():
            total_randoms = stack[key][j]
            total_randoms_ra = stack_ra[key][j]
            total_indices = np.arange(len(stack[key][j]))
            new_sample = random.choices(total_indices, k=Ns[key][j]) # Return a k sized list of elements chosen from the population with replacement.
            sample_medians[key][j].append(np.nanmedian(total_randoms[new_sample]))
            sample_medians_ra[key][j].append(np.nanmedian(total_randoms_ra[new_sample]))

    if i%100==0:
        logging.info('Finished {}.'.format(i))
    
names = {EXTENSION: 'err_median_troughsub_bootstrap',
        'flux_troughsub_2': 'err_median_troughsub_2_bootstrap',
        'red_cont_flux': 'err_median_redcontinuum_bootstrap'}
for key in stack.keys():
    mean_of_medians = [np.nanmean(sample_medians[key][j]) for j in range(M)]
    std_of_medians = [np.nanstd(sample_medians[key][j]) for j in range(M)]
    std_of_medians_ra = [np.nanstd(sample_medians_ra[key][j]) for j in range(M)]

    big_tab_proper[names[key]] = std_of_medians
    big_tab_proper[names[key]+'_ra'] = std_of_medians_ra

ascii.write(big_tab_proper, os.path.join(final_dir, f"radial_profiles_proper_multimasks_new_mask{DIR_APX}_unflagged.tab"))

