#import multiprocessing
import time
from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale, biweight_location

from astropy.cosmology import FlatLambdaCDM
# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import argparse

import sys
import os

from multiprocessing import Pool

#### Find out which switches are turned on or off. ########################

parser = argparse.ArgumentParser()
parser.add_argument("-l","--local", type=str, default="False",
                   help="Use local sky subtraction?")
parser.add_argument("-nf", "--newflag", type=str, default="True",
		help="Use Karl's new flagging method.")
parser.add_argument("-sr", "--subtract_residual", type=str, default="False", help="Subtract average spectrum after masking continuum fibers.")
parser.add_argument("-bc", "--background_correction", type=str, default="False", help="Add wavelength-dependent background correction.")
parser.add_argument("-f", "--farout", type=str, default="False", help="Measure surface brightness out to 100''.")
parser.add_argument("-na", "--new_agns", type=str, default = "False", help='Save the 52 new AGNs.')
parser.add_argument("--museskysub", type=str, default = "False", help="Imitate MUSE's sky subtraction: subtract average spectrum in each IFU.")
parser.add_argument('--hdr3', type=str, default='False', help='Use HDR3 data.')

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

if args.museskysub == "True":
	MUSE_SKYSUB = True
	DIR_APX = DIR_APX + "_MUSE"
	print("Imitating MUSE's sky subtraction: subtracting average spectrum in each IFU.")
elif args.museskysub == "False":
	MUSE_SKYSUB = False
	print("Not applying MUSE's sky subtraction.")
else:
	print("Could not identify a valid argument for museskysub: True or False.")

if args.hdr3 == 'True':
	HDR3 = True
	DIR_APX = DIR_APX + '_hdr3'
	print('Using HDR3')
elif args.hdr3 == 'False':
	HDR3 = False
	print('Using HDR2.1')
else:
	print('Wrong argument for hdr3: True or False.')

print("Final directory appendix: "+DIR_APX)


########## Define some functions #############################

def load_shot(shot):
	if HDR3:
		fileh = open_shot_file(shot, survey='hdr3')
	else:
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
	rs_kpc = (rs * cosmo.kpc_proper_per_arcmin(lae_redshift)).to(u.kpc)
	spec_here = spec_here[order]
	err_here = err_here[order]
	mask_7_here = mask_7_here[order]
	mask_10_here = mask_10_here[order]
	lae_ra = lae_ra[order]
	lae_dec = lae_dec[order]

	MEDFILT_CONTSUB = False #True
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
		new_mask = {}
		for cutoff in new_masks[0].keys():
			new_mask[cutoff] = new_masks[0][cutoff] * new_masks[1][cutoff]
			# make sure that the central 2'' are unmasked:
			new_mask[cutoff][rs <= 2.0 * u.arcsec] = 1
		print("{:.1f}% unmasked with {}% cutoff for LAE {}.".format(100*sum(new_mask[5])/len(new_mask[5]),5, detectid))
		print("Are the masks the same?", np.unique(new_mask[0] == new_mask[10]))

	# subtract the "continuum" within the absorption troughs (within 45AA of the line center, but not the inner 2xFWHM)
	lae_wave = lae["wave"]
	lae_linewidth = lae["linewidth"]
	cont_wlhere = (abs(def_wave - lae_wave) <= 40) & (abs(def_wave - lae_wave)>2.5*lae_linewidth)
	trough_continuum = np.nanmedian(spec_here[:,cont_wlhere], axis=1)
	N = np.nansum(ones[:,cont_wlhere], axis=1)
	trough_continuum_error = biweight_scale(spec_here[:,cont_wlhere], axis=1) / np.sqrt(N)
	trough_contsub = (spec_here.T - trough_continuum).T
	trough_contsub_error = np.sqrt(err_here.T**2 + trough_continuum_error**2).T
	
	cont_wlhere_2 = (def_wave > lae_wave) * (abs(def_wave - lae_wave) <= 40) & (abs(def_wave - lae_wave)>2.5*lae_linewidth)
	trough_continuum_2 = np.nanmedian(spec_here[:,cont_wlhere_2], axis=1)
	N = np.nansum(ones[:,cont_wlhere_2], axis=1)
	trough_continuum_error_2 = biweight_scale(spec_here[:,cont_wlhere_2], axis=1) / np.sqrt(N)
	trough_contsub_2 = (spec_here.T - trough_continuum_2).T
	trough_contsub_error_2 = np.sqrt(err_here.T**2 + trough_continuum_error_2**2).T


	# save core spectrum and halo spectra
	rest_wavelength = def_wave / (1 + lae_redshift)
	for r_min, r_max in zip(r_bins_kpc, r_bins_max_kpc):

		core_spectrum = np.nanmedian(spec_here[mask_7_here & (rs_kpc > r_min*u.kpc) & (rs_kpc <= r_max*u.kpc)], axis=0)
		core_spectrum_uf = np.nanmedian(spec_here[(rs_kpc > r_min*u.kpc) & (rs_kpc <= r_max*u.kpc)], axis=0)
		core_spectrum_contsub = np.nanmedian(continuum_subtracted[(rs_kpc > r_min*u.kpc) & (rs_kpc <= r_max*u.kpc)], axis=0)
		core_spectrum_trough_contsub = np.nanmedian(trough_contsub[(rs_kpc > r_min*u.kpc) & (rs_kpc <= r_max*u.kpc)], axis=0)

		save_tab = {"wave_rest": rest_wavelength,
			"central": spec_here[0], # central fiber
			"spec": core_spectrum,
			"spec_uf": core_spectrum_uf,
			"spec_contsub_uf": core_spectrum_contsub,
			"spec_troughsub_uf": core_spectrum_trough_contsub}

		path = os.path.join(savedir, f"core_spectra_unflagged{DIR_APX}/{int(r_min)}_{int(r_max)}")
		if not os.path.exists(path):
			os.mkdir(path)
			print("Creating path", path)
		save_name = os.path.join(savedir, f"core_spectra_unflagged{DIR_APX}/{int(r_min)}_{int(r_max)}/lae_{detectid}.dat")
		ascii.write(save_tab, save_name)

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
	spec_troughsub_sum_2 = np.nansum(trough_contsub_2[:,wlhere], axis=1)
	spec_troughsub_err_2 = np.sqrt(np.nansum(trough_contsub_error_2[:,wlhere], axis=1))

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
    
	red_cont_wlhere = (def_wave > lae_wave + 5*lae_linewidth) * (def_wave <= lae_wave + 5*lae_linewidth + 100)
	red_cont_flux = np.nanmedian(spec_here[:,red_cont_wlhere], axis=1)
    
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
	spec_troughsub_sum_2 = spec_troughsub_sum_2[mask].data[:] # troughsub with larger free window around Lya
	spec_troughsub_sum_4 = spec_troughsub_sum_4[mask].data[:]
	spec_troughsub_sum_11 = spec_troughsub_sum_11[mask].data[:]
	spec_troughsub_err = spec_troughsub_err[mask].data[:]
	spec_troughsub_err_2 = spec_troughsub_err_2[mask].data[:]
	spec_troughsub_err_4 = spec_troughsub_err_4[mask].data[:]
	spec_troughsub_err_11 = spec_troughsub_err_11[mask].data[:]
	mask_7_here_0 = mask_7_here[mask]
	mask_10_here_0 = mask_10_here[mask]
	lae_ra_0 = lae_ra[mask]
	lae_dec_0 = lae_dec[mask]
	red_cont_flux = red_cont_flux[mask].data[:]

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
		"flux_troughsub_2": spec_troughsub_sum_2,
		"flux_troughsub_4": spec_troughsub_sum_4,
		"flux_troughsub_11": spec_troughsub_sum_11,
		"err_troughsub": spec_troughsub_err,
		"err_troughsub_2": spec_troughsub_err_2,
		"err_troughsub_4": spec_troughsub_err_4,
		"err_troughsub_11": spec_troughsub_err_11,
		"sigma": err_sum,
		"sigma_4": err_sum_4,
		"sigma_11": err_sum_11,
		"mask_7": mask_7_here_0,
		"mask_10": mask_10_here_0,
		"red_cont_flux": red_cont_flux}
	if NEWFLAG:
		for cutoff in new_mask.keys():
			tab["new_mask_{}".format(cutoff)] = new_mask_0[cutoff]
	save_file = os.path.join(savedir, f"radial_profiles/laes{DIR_APX}/lae_{detectid}.dat")
	ascii.write(tab, save_file)
	print("Wrote to "+save_file)

	for d_wl in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
			-10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90, -95, -100, -105]:
		lae_wave = lae["wave"] + d_wl*2 # convert pixel to angstrom
		lae_linewidth = lae["linewidth"]

		# subtract the "continuum" within the absorption troughs (within 45AA of the line center, but not the inner 2xFWHM)
		cont_wlhere = (abs(def_wave - lae_wave) <= 40) & (abs(def_wave - lae_wave)>2.5*lae_linewidth)
		trough_continuum = np.nanmedian(spec_here[:,cont_wlhere], axis=1)
		N = np.nansum(ones[:,cont_wlhere], axis=1)
		trough_continuum_error = biweight_scale(spec_here[:,cont_wlhere], axis=1) / np.sqrt(N)
		trough_contsub = (spec_here.T - trough_continuum).T
		trough_contsub_error = np.sqrt(err_here.T**2 + trough_continuum_error**2).T
          
		cont_wlhere_2 = (abs(def_wave - lae_wave) <= 40) & (abs(def_wave - lae_wave)>3.5*lae_linewidth)
		trough_continuum_2 = np.nanmedian(spec_here[:,cont_wlhere_2], axis=1)
		N = np.nansum(ones[:,cont_wlhere_2], axis=1)
		trough_continuum_error_2 = biweight_scale(spec_here[:,cont_wlhere_2], axis=1) / np.sqrt(N)
		trough_contsub_2 = (spec_here.T - trough_continuum_2).T
		trough_contsub_error_2 = np.sqrt(err_here.T**2 + trough_continuum_error_2**2).T

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
		spec_troughsub_sum_2 = np.nansum(trough_contsub_2[:,wlhere], axis=1)
		spec_troughsub_err_2 = np.sqrt(np.nansum(trough_contsub_error_2[:,wlhere], axis=1))

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

		red_cont_wlhere = (def_wave > lae_wave + 5*lae_linewidth) * (def_wave <= lae_wave + 5*lae_linewidth + 100)
		red_cont_flux = np.nanmedian(spec_here[:,red_cont_wlhere], axis=1)
        
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
		spec_troughsub_sum_2 = spec_troughsub_sum_2[mask].data[:]
		spec_troughsub_sum_4 = spec_troughsub_sum_4[mask].data[:]
		spec_troughsub_sum_11 = spec_troughsub_sum_11[mask].data[:]
		spec_troughsub_err = spec_troughsub_err[mask].data[:]
		spec_troughsub_err_2 = spec_troughsub_err_2[mask].data[:]
		spec_troughsub_err_4 = spec_troughsub_err_4[mask].data[:]
		spec_troughsub_err_11 = spec_troughsub_err_11[mask].data[:]
		mask_7_here_0 = mask_7_here[mask]
		mask_10_here_0 = mask_10_here[mask]
		lae_ra_0 = lae_ra[mask]
		lae_dec_0 = lae_dec[mask]
		red_cont_flux = red_cont_flux[mask].data[:]

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
			"flux_troughsub_2": spec_troughsub_sum_2,
			"flux_troughsub_4": spec_troughsub_sum_4,
			"flux_troughsub_11": spec_troughsub_sum_11,
			"err_troughsub": spec_troughsub_err,
			"err_troughsub_2": spec_troughsub_err_2,
			"err_troughsub_4": spec_troughsub_err_4,
			"err_troughsub_11": spec_troughsub_err_11,
			"sigma": err_sum,
			"sigma_4": err_sum_4,
			"sigma_11": err_sum_11,
			"mask_7": mask_7_here_0,
			"mask_10": mask_10_here_0,
			"red_cont_flux": red_cont_flux}
        
		if NEWFLAG:
			for cutoff in new_mask_0.keys():
				tab["new_mask_{}".format(cutoff)] = new_mask_0[cutoff]
		save_file = os.path.join(savedir, f"radial_profiles/laes_wloffset{DIR_APX}/lae_{detectid}_{d_wl}.dat")
		ascii.write(tab, save_file)
		print("Wrote to "+save_file)

	return 1

basedir = "/work/05865/maja_n/stampede2/master"
savedir = "/scratch/05865/maja_n"

if args.new_agns == 'True':
	print('Saving 52 new AGNs.')
	complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "elixer_plots", "agn_unique_source_ids.txt"))
else:	   
	complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "high_sn_sources_combined.tab"))
	complete_lae_tab = complete_lae_tab[complete_lae_tab["mask"]==1]

order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

# include only high-S/N LAEs and exclude LAEs in large-residual areas.
#complete_lae_tab = complete_lae_tab[complete_lae_tab["sn"]>6.5]
#complete_lae_tab = complete_lae_tab[complete_lae_tab["wave"] - 1.5*complete_lae_tab["linewidth"] >3750]

core_spectra_path = os.path.join(savedir, f"core_spectra_unflagged{DIR_APX}")
lae_path = os.path.join(savedir, f"radial_profiles/laes{DIR_APX}")
lae_offset_path = os.path.join(savedir, f"radial_profiles/laes_wloffset{DIR_APX}")
for path in [core_spectra_path, lae_path, lae_offset_path]:
	if not os.path.exists(path):
		os.mkdir(path)
		print("Creating path", path)
	else:
		print("Path already exists", path)

# this is Dustin's list of LAEs
#complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "dustins_full_detids.dets"))

def_wave = np.arange(3470, 5542, 2.)
a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0)                                 
b = np.min([np.ones(1036)*1035, a+190], axis=0)                                             
c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0)
filter_min = np.array(c, dtype=int)
filter_max = np.array(b, dtype=int)

for shotid in np.unique(complete_lae_tab["shotid"]):

	#load the shot table and prepare full-frame sky subtracted spectra

	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	done = False #True 
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
	elif HDR3:
		ffskysub = shot_tab["calfib_ffsky"].copy()  # full frame sky subtraction
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

	if MUSE_SKYSUB:
		perc = 90
		for ifu in np.unique(shot_tab["ifuid"]):
			ifu_here = shot_tab["ifuid"] == ifu
			medians_lo = np.nanmedian(ffskysub[ifu_here][:,wlcont_lo], axis=1)
			medians_hi = np.nanmedian(ffskysub[ifu_here][:,wlcont_hi], axis=1)
			perc_lo = np.nanpercentile(medians_lo, perc)
			perc_hi = np.nanpercentile(medians_hi, perc)
			mask_here = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)
			sky_spectrum_here = np.nanmedian(ffskysub[ifu_here][mask_here], axis=0)
			ffskysub[ifu_here] -= sky_spectrum_here

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

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)

	with Pool(processes=8) as p:
		tmp = p.map(save_lae, laes_here["detectid"].data)	
	print(f"Finished {shotid}.")	
	continue

