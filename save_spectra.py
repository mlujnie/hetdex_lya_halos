#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import argparse

import sys
import os

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-l","--local", type=str, default="False",
                   help="Use local sky subtraction?")
args = parser.parse_args(sys.argv[1:])

if args.local == "False":
	LOCAL_SKYSUB = False
	DIR_APX = ""
	print("Using full-frame sky subtraction")
elif args.local == "True":
	LOCAL_SKYSUB = True
	DIR_APX = "_loc"
	print("Using local sky subtraction")
else:
	print("Could not identify a valid argument for local: True or False.")
	sys.exit

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
 
def save_lae(detectid):
	lae = complete_lae_tab[complete_lae_tab["detectid"]==detectid]
	lae_redshift = lae["wave"]/1215.67 - 1
	
	lae_ra, lae_dec = lae["ra"], lae["dec"]
	lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
	rs = lae_coords.separation(shot_coords)
	mask = rs <= 2*u.arcsec
	rs = rs[mask]
	spec_here = ffskysub[mask]
	err_here = spec_err[mask]
	mask_7_here = mask_7[mask]
	mask_10_here = mask_10[mask]
	lae_ra = shot_tab["ra"][mask]
	lae_dec = shot_tab["dec"][mask]
	order = np.argsort(rs)
	rs = rs[order]
	spec_here = spec_here[order]
	err_here = err_here[order]
	mask_7_here = mask_7_here[order]
	mask_10_here = mask_10_here[order]
	lae_ra = lae_ra[order]
	lae_dec = lae_dec[order]

	MEDFILT_CONTSUB = True

	if MEDFILT_CONTSUB:
		continuum = np.zeros(spec_here.shape)
		continuum_error = np.zeros(spec_here.shape)
		ones = np.ones(spec_here.shape)
		ones[~np.isfinite(spec_here)] = np.nan
		indices = np.arange(1036)
		for i in indices:
			idxhere = (indices >= filter_min[i])&(indices <= filter_max[i])
			continuum[:,i] += np.nanmedian(spec_here[:,idxhere], axis=1)
			N = np.nansum(ones[:,idxhere], axis=1)
			continuum_error[:,i] += biweight_scale(spec_here[:,idxhere], axis=1, ignore_nan=True)/np.sqrt(N)
		continuum[continuum==0.0] = np.nan
		continuum_subtracted = spec_here.copy() - continuum
		continuum_subtracted_error = np.sqrt(err_here**2 + continuum_error**2)
		continuum_subtracted[continuum_subtracted==0.0] = np.nan
	else:
		continuum = []
		for i in range(len(spec_here)):
			here = np.isfinite(spec_here[i])
			if len(here[here]) < 1:
				continuum.append(np.nan * def_wave)
				continue
			ps = np.polyfit(def_wave[here], spec_here[i][here], 5)
			continuum.append(ps[0] * def_wave**5 + ps[1] * def_wave**4 + ps[2] * def_wave**3 + ps[3] * def_wave**2 + ps[4] * def_wave + ps[5])
		continuum = np.array(continuum)
		continuum_subtracted = spec_here.copy() - continuum
		continuum_subtracted[continuum_subtracted==0.0] = np.nan

	# subtract the "continuum" within the absorption troughs (within 45AA of the line center, but not the inner 2xFWHM)
	lae_wave = lae["wave"]
	lae_linewidth = lae["linewidth"]
	cont_wlhere = (abs(def_wave - lae_wave) <= 45) & (abs(def_wave - lae_wave)>2.5*lae_linewidth)
	trough_continuum = np.nanmedian(spec_here[:,cont_wlhere], axis=1)
	N = np.nansum(ones[:,cont_wlhere], axis=1)
	trough_continuum_error = biweight_scale(spec_here[:,cont_wlhere], axis=1) / np.sqrt(N)
	print("spec_here.shape, trough_continuum.shape", spec_here.shape, trough_continuum.shape)
	trough_contsub = (spec_here.T - trough_continuum).T
	trough_contsub_error = np.sqrt(err_here.T**2 + trough_continuum_error**2).T
	

	# save core spectrum and halo spectra
	rest_wavelength_lae = def_wave / (1 + lae_redshift)
	core_spectrum = np.nanmedian(spec_here[mask_7_here & (rs <= 2*u.arcsec)], axis=0)
	core_spectrum_uf = np.nanmedian(spec_here[(rs <= 2*u.arcsec)], axis=0)
	spectrum_0_2_contsub = np.nanmedian(continuum_subtracted[(rs <= 2*u.arcsec)], axis=0)
	spectrum_0_2_trough_contsub = np.nanmedian(trough_contsub[(rs <= 2*u.arcsec)], axis=0)

	save_tab = {"wave_rest_lae": rest_wavelength_lae,
			"central": spec_here[0], # central fiber
			"spec_0_2": core_spectrum,
			"spec_uf_0_2": core_spectrum_uf,
			"spec_contsub_uf_0_2": spectrum_0_2_contsub,
			"spec_troughsub_uf_0_2": spectrum_0_2_trough_contsub}
	save_name = os.path.join(savedir, f"nearby_spectra_unflagged{DIR_APX}/gal_{detectid}.dat")
	ascii.write(save_tab, save_name)
	print("Wrote to ", save_name)
	return 1

   
basedir = "/work2/05865/maja_n/stampede2/master"
savedir = "/scratch/05865/maja_n"
complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "oii_sample.tab"))
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

# include only high-S/N LAEs and exclude LAEs in large-residual areas.
#complete_lae_tab = complete_lae_tab[complete_lae_tab["sn"]>6.5]
#complete_lae_tab = complete_lae_tab[complete_lae_tab["wave"] - 1.5*complete_lae_tab["linewidth"] >3750]

# this is Dustin's list of LAEs
#complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "dustins_full_detids.dets"))

def_wave = np.arange(3470, 5542, 2.)
a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0)                                 
b = np.min([np.ones(1036)*1035, a+190], axis=0)                                             
c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0)
filter_min = np.array(c, dtype=int)
filter_max = np.array(b, dtype=int)

for shotid in np.unique(complete_lae_tab["shotid"])[:]:

	#load the shot table and prepare full-frame sky subtracted spectra

	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	done = True 
	for detectid in laes_here["detectid"].data:
		done *= os.path.exists(os.path.join(savedir, f"core_spectra_unflagged{DIR_APX}/lae_{detectid}.dat"))
	if done:
		print("Already finished", shotid)
		continue
	try:
		shot_tab = load_shot(shotid)
	except Exception as e:
		print(f"Could not load shot {shotid}. Error message:")
		print(e)
		continue

	if LOCAL_SKYSUB:
		ffskysub = shot_tab["calfib"].copy() # local sky subtraction
	else:
		ffskysub = shot_tab["spec_fullsky_sub"].copy()  # full frame sky subtraction
	ffskysub[ffskysub==0] = np.nan

	# mask sky lines and regions with large residuals
	for l_min, l_max in [(5455,5470), (5075,5095), (4355,4370)]: # only sky lines, not galactic lines.
		#[(3720, 3750), (4850,4870), (4950,4970),(5000,5020)]: # these are galactic emission lines that don't matter for HETDEX
		wlhere = (def_wave >= l_min) & (def_wave <= l_max)
		ffskysub[:,wlhere] = np.nan

	# exclude extreme continuum values
	perc = 93
	
	wlcont_lo = (def_wave > 4000)&(def_wave <= 4500)
	medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1)
	perc_lo = np.nanpercentile(medians_lo, perc)

	wlcont_hi = (def_wave > 4800)&(def_wave <= 5300)
	medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1)
	perc_hi = np.nanpercentile(medians_hi, perc)
	mask_7 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)

	perc = 90
	
	wlcont_lo = (def_wave > 4000)&(def_wave <= 4500)
	medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1)
	perc_lo = np.nanpercentile(medians_lo, perc)

	wlcont_hi = (def_wave > 4800)&(def_wave <= 5300)
	medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1)
	perc_hi = np.nanpercentile(medians_hi, perc)
	mask_10 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)

	spec_err = shot_tab["calfibe"].copy()
	spec_err[~np.isfinite(ffskysub)] = np.nan

	# subtract the median spectrum (residual)
	residual = np.nanmedian(ffskysub[mask_7], axis=0)
	ffskysub = ffskysub - residual

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)

	with Pool(processes=8) as p:
		tmp = p.map(save_lae, laes_here["detectid"].data)	
	print(f"Finished {shotid}.")	
	continue
