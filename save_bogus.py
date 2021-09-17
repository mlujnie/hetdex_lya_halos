#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import random
from multiprocessing import Pool

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale, biweight_location

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import argparse

import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l","--local", type=str, default="False",
                   help="Use local sky subtraction?")
parser.add_argument("-nf", "--newflag", type=str, default="True",
		help="Use Karl's new flagging method.")
parser.add_argument("-sr", "--subtract_residual", type=str, default="False", help="Subtract average spectrum after masking continuum fibers.")
parser.add_argument("-bc", "--background_correction", type=str, default="False", help="Add wavelength-dependent background correction.")
parser.add_argument("-f", "--farout", type=str, default="False", help="Measure surface brightness out to 100''.")

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
	sys.exit()

if args.newflag == "True":
	print("Using Karl's new flagging method.")
	DIR_APX = "_newflag"+DIR_APX
	NEWFLAG = True
elif args.newflag == "False":
	print("Not flagging fibers.")
	NEWFLAG = False
else:
	print("Could not identify a valid argument for newflag: True or False.")
	sys.exit()

if args.subtract_residual == "True":
	print("Subtracting residual.")
	DIR_APX = DIR_APX + "_ressub"
	SUBTRACT_RESIDUAL = True
elif args.subtract_residual == "False":
	print("Not subtracting shot residual.")
	SUBTRACT_RESIDUAL = False
else:
	print("Could not identify a valid argument for subtract_residual: True or False.")
	sys.exit()

if args.background_correction == "True":
	BACKGROUND_CORRECTION = True
	DIR_APX = DIR_APX + "_backcor"
	print("Applying background correction.")
elif args.background_correction == "False":
	BACKGROUND_CORRECTION = False
	print("Not applying background correction.")
else:
	print("Could not identify a valid argument for background_correction: True or False.")

if args.farout == "True":
	FAR_OUT = True
	DIR_APX = DIR_APX + "_100"
	print("Going out to 100''.")
elif args.farout == "False":
	FAR_OUT = False
	print("Going out to 20''.")
else:
	print("Could not identify a valid argument for farout: True or False.")
print("Final directory appendix: "+DIR_APX)


def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
 
def save_lae(detectid):
	lae = complete_lae_tab[complete_lae_tab["detectid"]==detectid]
	
	for iternum in range(3):
		randnum = random.randint(0, len(shot_tab)-1)
		d_ra = random.uniform(-1, 1)/3600.
		d_dec = random.uniform(-1, 1)/3600.
		d_wave = random.uniform(-50, 50)

		lae_ra, lae_dec = shot_tab["ra"][randnum]+d_ra, shot_tab["dec"][randnum]+d_dec

		lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
		rs = lae_coords.separation(shot_coords)
		if FAR_OUT:
			mask = rs <= 100*u.arcsec
		else:
			mask = rs <= 20*u.arcsec
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

		MEDFILT_CONTSUB = False
		NO_CONTSUB = True

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
		elif NO_CONTSUB:
			continuum_subtracted = np.zeros(spec_here.shape)
			continuum_subtracted_error = np.ones(spec_here.shape)
			ones = np.ones(spec_here.shape)
			ones[~np.isfinite(spec_here)] = np.nan
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
		
		if NEWFLAG:
			wlhere_low = (def_wave >= 3700) * (def_wave <= 4200)
			wlhere_high = (def_wave >= 4800) * (def_wave <= 5300)
			fibers_here = rs > 2.0 * u.arcsec
			new_masks = []
			for wl_mask in [wlhere_low, wlhere_high]:
				try:
					new_mask_tmp = np.ones(len(spec_here), dtype=int)

					averages = biweight_location(spec_here[:,wl_mask], axis=1, ignore_nan = True)
					av_average = biweight_location(averages[fibers_here], ignore_nan=True)
					av_std = biweight_scale(averages[fibers_here], ignore_nan=True)
					outliers = abs(averages - av_average) > 3*av_std
					new_mask_tmp[outliers] = 0
					print("{:.1f}% unmasked after first step for LAE {}.".format(100*sum(new_mask_tmp)/len(new_mask_tmp), detectid))
					
					new_mask_dict = {}
					for cutoff in [0, 5, 10]:
						new_mask_dict[cutoff] = np.ones(len(spec_here), dtype=int)
						new_mask_dict[cutoff][new_mask_tmp == 0] = 0
						perc = np.nanpercentile(averages[fibers_here * new_mask_tmp], 100-cutoff)
						print(cutoff, perc)
						highest = averages > perc
						new_mask_dict[cutoff][highest] = 0
					new_mask_dict[0] = np.ones(len(spec_here), dtype=int)
					new_mask_dict[0][new_mask_tmp == 0] = 0
					
					new_masks.append(new_mask_dict)
				except Exception as e:
					print("An error occurred, adding False flags.")
					print(e)
					new_mask_tmp = np.zeros(len(spec_here), dtype=int)
					new_mask_dict = {0: new_mask_tmp, 5: new_mask_tmp, 10: new_mask_tmp}
					new_masks.append(new_mask_dict)
			new_mask = {}
			for cutoff in new_masks[0].keys():
				new_mask[cutoff] = new_masks[0][cutoff] * new_masks[1][cutoff]
				# make sure that the central 2'' are unmasked:
				new_mask[cutoff][rs <= 2.0 * u.arcsec] = 1
			print("{:.1f}% unmasked with {}% cutoff for LAE {}.".format(100*sum(new_mask[5])/len(new_mask[5]),5, detectid))
			print("Are the masks the same?", np.unique(new_mask[0] == new_mask[10]))


		lae_wave = lae["wave"] + d_wave
		lae_linewidth = lae["linewidth"]
		cont_wlhere = (abs(def_wave - lae_wave) <= 40) & (abs(def_wave - lae_wave)>2.5*lae_linewidth)
		trough_continuum = np.nanmedian(spec_here[:,cont_wlhere], axis=1)
		N = np.nansum(ones[:,cont_wlhere], axis=1)
		trough_continuum_error = biweight_scale(spec_here[:,cont_wlhere], axis=1) / np.sqrt(N)
		trough_contsub = (spec_here.T - trough_continuum).T
		trough_contsub_error = np.sqrt(err_here.T**2 + trough_continuum_error**2).T
		
		wlhere = abs(def_wave - lae_wave) <= 1.5 * lae_linewidth
		wlhere_4 = abs(def_wave - lae_wave) <= 2 # integration window 4AA
		wlhere_11 = abs(def_wave - lae_wave) <= 11/2. # integration window 11AA
		
		# variable integration window	
		spec_sum = np.nansum(spec_here[:,wlhere], axis=1)
		err_sum = np.sqrt(np.nansum(err_here[:,wlhere]**2, axis=1))
		spec_sub_sum = np.nansum(continuum_subtracted[:,wlhere], axis=1)
		spec_sub_err = np.sqrt(np.nansum(continuum_subtracted_error[:,wlhere]**2, axis=1))
		spec_troughsub_sum = np.nansum(trough_contsub[:,wlhere], axis=1)
		spec_troughsub_err = np.sqrt(np.nansum(trough_contsub_error[:,wlhere], axis=1))

		# fixed integration window (4AA)			
		spec_sum_4 = np.nansum(spec_here[:,wlhere_4], axis=1)
		err_sum_4 = np.sqrt(np.nansum(err_here[:,wlhere_4]**2, axis=1))
		spec_sub_sum_4 = np.nansum(continuum_subtracted[:,wlhere_4], axis=1)
		spec_sub_err_4 = np.sqrt(np.nansum(continuum_subtracted_error[:,wlhere_4]**2, axis=1))
		spec_troughsub_sum_4 = np.nansum(trough_contsub[:,wlhere_4], axis=1)
		spec_troughsub_err_4 = np.sqrt(np.nansum(trough_contsub_error[:,wlhere_4], axis=1))
					
		# fixed integration window (11AA)
		spec_sum_11 = np.nansum(spec_here[:,wlhere_11], axis=1)
		err_sum_11 = np.sqrt(np.nansum(err_here[:,wlhere_11]**2, axis=1))
		spec_sub_sum_11 = np.nansum(continuum_subtracted[:,wlhere_11], axis=1)
		spec_sub_err_11 = np.sqrt(np.nansum(continuum_subtracted_error[:,wlhere_11]**2, axis=1))
		spec_troughsub_sum_11 = np.nansum(trough_contsub[:,wlhere_11], axis=1)
		spec_troughsub_err_11 = np.sqrt(np.nansum(trough_contsub_error[:,wlhere_11], axis=1))

		mask = (spec_sum != 0) & (err_sum != 0) & (spec_sum_4 != 0) & (err_sum_4 != 0) & (spec_sum_11 != 0 ) & (err_sum_11 != 0) & (spec_troughsub_sum != 0) & (spec_troughsub_sum_4 != 0) & (spec_troughsub_sum_11 != 0)
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
		spec_sub_err = spec_sub_err[mask].data[:]
		spec_sub_err_4 = spec_sub_err_4[mask].data[:]
		spec_sub_err_11 = spec_sub_err_11[mask].data[:]
		spec_troughsub_sum = spec_troughsub_sum[mask].data[:]
		spec_troughsub_sum_4 = spec_troughsub_sum_4[mask].data[:]
		spec_troughsub_sum_11 = spec_troughsub_sum_11[mask].data[:]
		spec_troughsub_err = spec_troughsub_err[mask].data[:]
		spec_troughsub_err_4 = spec_troughsub_err_4[mask].data[:]
		spec_troughsub_err_11 = spec_troughsub_err_11[mask].data[:]
		mask_7_here_0 = mask_7_here[mask]
		mask_10_here_0 = mask_10_here[mask]
		lae_ra_0 = lae_ra[mask]
		lae_dec_0 = lae_dec[mask]
			
		if NEWFLAG:
			new_mask_0 = {}
			for cutoff in new_mask.keys():
				new_mask_0[cutoff] = new_mask[cutoff][mask]
		
		tab = {"r": rs_0,
			"ra": lae_ra_0,
			"dec": lae_dec_0,
			"flux":spec_sum,
			"flux_4":spec_sum_4,
			"flux_11":spec_sum_11,
			"flux_contsub":spec_sub_sum,
			"flux_contsub_4":spec_sub_sum_4,
			"flux_contsub_11":spec_sub_sum_11,
			"err_contsub": spec_sub_err,
			"err_contsub_4": spec_sub_err_4,
			"err_contsub_11": spec_sub_err_11,
			"flux_troughsub": spec_troughsub_sum,
			"flux_troughsub_4": spec_troughsub_sum_4,
			"flux_troughsub_11": spec_troughsub_sum_11,
			"err_troughsub": spec_troughsub_err,
			"err_troughsub_4": spec_troughsub_err_4,
			"err_troughsub_11": spec_troughsub_err_11,
			"sigma": err_sum,
			"sigma_4": err_sum_4,
			"sigma_11": err_sum_11,
			"mask_7": mask_7_here_0,
			"mask_10": mask_10_here_0}
		if NEWFLAG:
			for cutoff in new_mask.keys():
				tab["new_mask_{}".format(cutoff)] = new_mask_0[cutoff]

		save_file = os.path.join(savedir, f"radial_profiles/bogus_laes{DIR_APX}/bogus_{iternum}_{detectid}.dat")
		ascii.write(tab, save_file)
		print("Wrote to "+save_file)
	return 1

basedir = "/work2/05865/maja_n/stampede2/master"
savedir = "/scratch/05865/maja_n"
complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "high_sn_sources.tab"))
complete_lae_tab = complete_lae_tab[complete_lae_tab["mask"]==1]
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

# include only high-S/N LAEs and exclude LAEs in large-residual areas.
#complete_lae_tab = complete_lae_tab[complete_lae_tab["sn"]>6.5]
#complete_lae_tab = complete_lae_tab[complete_lae_tab["wave"] - 1.5*complete_lae_tab["linewidth"] >3750]

path = os.path.join(savedir, f"radial_profiles/bogus_laes{DIR_APX}")
if not os.path.exists(path):
	os.mkdir(path)
	print("Creating path", path)
else:
	print("Path already exists", path)


def_wave = np.arange(3470, 5542, 2.)
a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0)                                 
b = np.min([np.ones(1036)*1035, a+190], axis=0)                                             
c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0)
filter_min = np.array(c, dtype=int)
filter_max = np.array(b, dtype=int)

for shotid in np.unique(complete_lae_tab["shotid"])[::-1]:

	#load the shot table and prepare full-frame sky subtracted spectra

	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	if len(laes_here)<1:
		continue
	done = True 
	for detectid in laes_here["detectid"].data:
		done *= os.path.exists(os.path.join(savedir, f"radial_profiles/bogus_laes{DIR_APX}/bogus_1_{detectid}.dat"))

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
		ffskysub = shot_tab["calfib"].copy()
	else:
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

	if SUBTRACT_RESIDUAL:
		# subtract the median spectrum (residual)
		residual = np.nanmedian(ffskysub[mask_7], axis=0)
		ffskysub = ffskysub - residual

	if BACKGROUND_CORRECTION:
		if SUBTRACT_RESIDUAL:
			xmid = 4800
			ymid = 0.0035
			y0, y1 = 0.015, 0.0025
		else:
			xmid = 4800
			ymid = 0.004
			y0, y1 = 0.015, 0.0025
		# add a wavelength-dependent background model to the data
		background = np.where(def_wave < xmid, (def_wave-3500)*(ymid-y0)/(xmid-3500) + y0, (def_wave-xmid)*(y1-ymid)/(5500-xmid) + ymid)
		ffskysub = ffskysub + background

	spec_err = shot_tab["calfibe"].copy()
	spec_err[~np.isfinite(ffskysub)] = np.nan

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)


	with Pool(processes=8) as p:
		tmp = p.map(save_lae, laes_here["detectid"].data)	
	print(f"Finished {shotid}.")	
	continue

