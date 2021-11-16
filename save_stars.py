#import multiprocessing
import time
import random
from hetdex_tools.get_spec import get_spectra

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import biweight_scale
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)

from hetdex_api.shot import *

from astropy.io import ascii
import glob

import sys
import os

from multiprocessing import Pool

def load_shot(shot):
        fileh = open_shot_file(shot)
        table = Table(fileh.root.Data.Fibers.read())
        fileh.close()
        return table
 
def save_star(detectid):
	try:
		lae = complete_lae_tab[complete_lae_tab["detectid"]==detectid]
		lae_ra, lae_dec = lae["ra"], lae["dec"]
		lae_coords = SkyCoord(ra = lae_ra*u.deg, dec = lae_dec*u.deg)
		rs = lae_coords.separation(shot_coords).arcsec
		mask = rs <= 100

		if len(mask[mask]) < 10:
			print("{} is empty.".format(detectid))
			return 0

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

		wlhere = (def_wave > 4550) & (def_wave <= 4650)
			
		weights = err_here[:,wlhere] ** (-2)
		weights_sum = np.nansum(weights, axis=1)
		flux_mean = np.nansum(spec_here[:,wlhere]*weights, axis=1) / weights_sum
		flux_mean_error = 1./np.sqrt(weights_sum)

		flux_minus_cont, flux_minus_cont_error = {}, {}

		for central_lambda in [3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400]:
			#central_lambda = random.choice(sources['wave']) # choose a random central wavelength from the LAE sample.

			loc_cont_wlhere = (abs(def_wave - central_lambda) <= 40) * (abs(def_wave-central_lambda)>17/2.)
			local_continuum_median = np.nanmedian(spec_here[:,loc_cont_wlhere], axis=1)
			ones = np.ones(spec_here.shape)
			ones[~np.isfinite(spec_here)] = 0
			N = np.nansum(ones[:,loc_cont_wlhere], axis=1)
			local_continuum_error = np.nanstd(spec_here[:,loc_cont_wlhere], axis=1)/np.sqrt(N)

			loc_wlhere = abs(def_wave - central_lambda) <= 10/2.
			spec_minus_median_cont = (spec_here.T - local_continuum_median).T
			flux_minus_cont[central_lambda] = np.nansum(spec_minus_median_cont[:,loc_wlhere], axis=1)
			flux_minus_cont_error[central_lambda] = np.sqrt(flux_mean_error**2 + local_continuum_error**2)
		
		mask = (flux_mean != 0) & (flux_mean_error != 0)
		rs_0 = rs[mask][:] #/ u.arcsec
		#rs_0 = rs_0.decompose()

		flux_mean = flux_mean[mask].data[:]
		flux_mean_error = flux_mean_error[mask].data[:]
		mask_7_here_0 = mask_7_here[mask]
		mask_10_here_0 = mask_10_here[mask]
		lae_ra_0 = lae_ra[mask]
		lae_dec_0 = lae_dec[mask]

		for cl in flux_minus_cont.keys():
			flux_minus_cont_error[cl] = flux_minus_cont_error[cl][mask].data[:]
			flux_minus_cont[cl] = flux_minus_cont[cl][mask].data[:]
			
		tab = {"r": rs_0,
			"ra": lae_ra_0,
			"dec": lae_dec_0,
			"flux": flux_mean,
			"sigma": flux_mean_error,
			"mask_7": mask_7_here_0,
			"mask_10": mask_10_here_0}
		for cl in flux_minus_cont.keys():
			tab['flux_contsub_{}'.format(cl)] = flux_minus_cont[cl]
			tab['flux_contsub_error_{}'.format(cl)] = flux_minus_cont_error[cl]
		save_file = os.path.join(basedir, f"radial_profiles/stars_skymask/star_{detectid}.dat")
		ascii.write(tab, save_file)
		print("Wrote to "+save_file)
	except Exception as e:
		print("{} failed: ".format(detectid))
		print(e)
		return 0
	return 1

   
basedir = "/work/05865/maja_n/stampede2/master"
complete_lae_tab = ascii.read(os.path.join(basedir, "karls_suggestion", "star_gaia_tab.tab"))
sources = ascii.read(os.path.join(basedir, "karls_suggestion", "high_sn_sources_combined.tab"))
sources = sources[(sources['mask']==1)*(sources['sn']>6.5)]
# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
sources["redshift"] = z_lae
sources["luminosity"] = (sources["flux_213"])*1e-17*4*np.pi*(cosmo.luminosity_distance(sources["redshift"]).to(u.cm)/u.cm)**2
c = 3*10**5 # km/s
doppler_v_of = c * sources["linewidth"] / sources["wave"]
sources["linewidth_km/s"] = doppler_v_of
sources = sources[(sources['linewidth_km/s']<1000/2.35)*(sources['luminosity']<1e43)]

#compddlete_lae_tab = complete_lae_tab[complete_lae_tab["mask"]==1]
order = np.argsort(complete_lae_tab["shotid"])
complete_lae_tab = complete_lae_tab[order]

def_wave = np.arange(3470, 5542, 2.)
t_oldest = os.path.getmtime(basedir+"/radial_profiles/laes/lae_2100640938.dat")

for shotid in np.unique(complete_lae_tab["shotid"]):
	#load the shot table and prepare full-frame sky subtracted spectra
	laes_here = complete_lae_tab[complete_lae_tab["shotid"]==shotid]
	done = False #True
	if done:
		t_oldest = os.path.getmtime(os.path.join(basedir, "radial_profiles/stars_skymask/star_1596349424488144256.dat"))
		for detectid in laes_here["detectid"].data:
			done *= os.path.exists(os.path.join(basedir, "radial_profiles/stars_skymask/star_{}.dat".format(detectid)))
			if done:
				t = os.path.getmtime(os.path.join(basedir, "radial_profiles/stars_skymask/star_{}.dat".format(detectid)))
				done *= t > t_oldest
			#done *= os.path.exists(os.path.join(basedir, f"radial_profiles/stars_skymask/star_{detectid}.dat")) #t >= t_oldest # CHANGE THIS!!!
		#os.path.exists(os.path.join(basedir, f"radial_profiles/laes/lae_{detectid}.dat"))
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
	for l_min, l_max in [(5455,5470), (5075,5095), (4355,4370)]: #(3720, 3750), (4850,4870), (4950,4970),(5000,5020),(5455,5470),(5075,5095),(4355,4370)]:
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
	#ffskysub[abs(medians_lo)>perc_lo] *= np.nan
	#ffskysub[abs(medians_hi)>perc_hi] *= np.nan
	mask_7 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)

	perc = 90
	
	wlcont_lo = (def_wave > 4000)&(def_wave <= 4500)
	medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1)
	perc_lo = np.nanpercentile(medians_lo, perc)

	wlcont_hi = (def_wave > 4800)&(def_wave <= 5300)
	medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1)
	perc_hi = np.nanpercentile(medians_hi, perc)
	#ffskysub[abs(medians_lo)>perc_lo] *= np.nan
	#ffskysub[abs(medians_hi)>perc_hi] *= np.nan
	mask_10 = (abs(medians_lo)<perc_lo) & (abs(medians_hi)<perc_hi)

	spec_err = shot_tab["calfibe"].copy()
	spec_err[~np.isfinite(ffskysub)] = np.nan

	shot_coords = SkyCoord(ra=shot_tab["ra"]*u.deg, dec=shot_tab["dec"]*u.deg)

	with Pool(processes=8) as p:
		tmp = p.map(save_star, laes_here["detectid"].data)	
	print(f"Finished {shotid}.")	

