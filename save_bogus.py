#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import random
from multiprocessing import Pool

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import sys
import os
from weighted_biweight import biweight_location_weights_karl

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
 
def save_lae(detectid):
	lae = complete_lae_tab[complete_lae_tab["detectid"]==detectid]
	
	for iternum in range(2):
		randnum = random.randint(0, len(shot_tab)-1)
		d_ra = random.uniform(-1, 1)/3600.
		d_dec = random.uniform(-1, 1)/3600.
		d_wave = random.uniform(-50, 50)

		lae_ra, lae_dec = shot_tab["ra"][randnum]+d_ra, shot_tab["dec"][randnum]+d_dec

		lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
		rs = lae_coords.separation(shot_coords)
		mask = rs <= 50*u.arcsec
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

		continuum = np.zeros(spec_here.shape)
		indices = np.arange(1036)
		for i in indices:
			idxhere = (indices >= filter_min[i])&(indices <= filter_max[i])
			continuum[:,i] += np.nanmedian(spec_here[:,idxhere], axis=1)
		continuum[continuum==0.0] = np.nan
		continuum_subtracted = spec_here.copy() - continuum
		continuum_subtracted[continuum_subtracted == 0.0] = np.nan
		
		lae_wave = lae["wave"] + d_wave
		lae_linewidth = lae["linewidth"]
		wlhere = abs(def_wave - lae_wave) <= 1.5 * lae_linewidth
		wlhere_4 = abs(def_wave - lae_wave) <= 2 # integration window 4AA
		wlhere_11 = abs(def_wave - lae_wave) <= 11/2. # integration window 11AA
		
		# variable integration window	
		spec_sum = np.nansum(spec_here[:,wlhere], axis=1)
		err_sum = np.sqrt(np.nansum(err_here[:,wlhere]**2, axis=1))

		spec_sub_sum = np.nansum(continuum_subtracted[:,wlhere], axis=1)
	
		# fixed integration window (4AA)			
		spec_sum_4 = np.nansum(spec_here[:,wlhere_4], axis=1)
		err_sum_4 = np.sqrt(np.nansum(err_here[:,wlhere_4]**2, axis=1))

		spec_sub_sum_4 = np.nansum(continuum_subtracted[:,wlhere_4], axis=1)
					
		# fixed integration window (11AA)
		spec_sum_11 = np.nansum(spec_here[:,wlhere_11], axis=1)
		err_sum_11 = np.sqrt(np.nansum(err_here[:,wlhere_11]**2, axis=1))

		spec_sub_sum_11 = np.nansum(continuum_subtracted[:,wlhere_11], axis=1)
	
		mask = (spec_sum != 0) & (err_sum != 0) & (spec_sum_4 != 0) & (err_sum_4 != 0) & (spec_sum_11 != 0 ) & (err_sum_11 != 0)
		rs_0 = rs[mask][:] / u.arcsec
		rs_0 = rs_0.decompose()

		spec_sum = spec_sum[mask].data[:]
		spec_sum_4 = spec_sum_4[mask].data[:]
		spec_sum_11 = spec_sum_11[mask].data[:]
		err_sum = err_sum[mask].data[:]
		err_sum_4 = err_sum_4[mask].data[:]
		err_sum_11 = err_sum_11[mask].data[:]
		spec_sub_sum = spec_sub_sum[mask].data[:]
		spec_sub_sum_4 = spec_sub_sum_4[mask].data[:]
		spec_sub_sum_11 = spec_sub_sum_11[mask].data[:]
		mask_7_here_0 = mask_7_here[mask]
		mask_10_here_0 = mask_10_here[mask]
		lae_ra_0 = lae_ra[mask]
		lae_dec_0 = lae_dec[mask]
		
		tab = {"r": rs_0,
			"ra": lae_ra_0,
			"dec": lae_dec_0,
			"flux":spec_sum,
			"flux_4":spec_sum_4,
			"flux_11":spec_sum_11,
			"flux_contsub":spec_sub_sum,
			"flux_contsub_4":spec_sub_sum_4,
			"flux_contsub_11":spec_sub_sum_11,
			"sigma": err_sum,
			"sigma_4": err_sum_4,
			"sigma_11": err_sum_11,
			"mask_7": mask_7_here_0,
			"mask_10": mask_10_here_0}
			
		save_file = os.path.join(basedir, f"radial_profiles/bogus_laes_skymask_loc/bogus_{iternum}_{detectid}.dat")
		ascii.write(tab, save_file)
	print("Wrote to "+save_file)
	return 1

   
basedir = "/work2/05865/maja_n/stampede2/master"
complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "high_sn_sources.tab"))
complete_lae_tab = complete_lae_tab[complete_lae_tab["mask"]==1]
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

def_wave = np.arange(3470, 5542, 2.)
a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0)                                 
b = np.min([np.ones(1036)*1035, a+190], axis=0)                                             
c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0)
filter_min = np.array(c, dtype=int)
filter_max = np.array(b, dtype=int)

for shotid in np.unique(complete_lae_tab["shotid"]):

	#load the shot table and prepare full-frame sky subtracted spectra

	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	if len(laes_here)<1:
		continue
	done = True 
	for detectid in laes_here["detectid"].data:
		done *= os.path.exists(os.path.join(basedir, f"radial_profiles/bogus_laes_skymask_loc/bogus_1_{detectid}.dat"))

	if done:
		print("Already finished", shotid)
		continue
	try:
		shot_tab = load_shot(shotid)
	except Exception as e:
		print(f"Could not load shot {shotid}. Error message:")
		print(e)
		continue
	ffskysub = shot_tab["spec_fullsky_sub"].copy()
	ffskysub[ffskysub==0] = np.nan

	# mask sky lines and regions with large residuals
	for l_min, l_max in [(5455,5470),(5075,5095),(4355,4370)]: # only sky lines, not galactic emission lines
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

	# exclude extreme continuum values
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

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)


	with Pool(processes=8) as p:
		tmp = p.map(save_lae, laes_here["detectid"].data)	
	print(f"Finished {shotid}.")	
	continue

