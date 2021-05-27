import astropy.units as u 
from astropy.coordinates import SkyCoord
from hetdex_api.shot import *
import numpy as np
from astropy.io import ascii
from astropy.stats import biweight_location, biweight_scale
import sys
import random

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
print(len(sources))
shot_ids = np.unique(sources["shotid"])
print(len(shot_ids))

#shot_ids_random = [shot_ids[random.randint(0, len(shot_ids))] for i in range(10)]
#shot_ids_random = np.unique(shot_ids_random)
shot_ids_random = [20190301015, 20190325015, 20190408032, 20190501026, 20190504020, 20190608020, 20190827018, 20191004017, 20191006027]
shot_ids_random = [20190302032]
print(shot_ids_random)

for shotid in shot_ids_random:
	fileh = open_shot_file(shotid)
	tab = fileh.root.Data.Fibers[:]
	ffskysub = tab["spec_fullsky_sub"].copy() 
	ffskysub[ffskysub==0] = np.nan

	def_wave = np.arange(3470, 5542, 2)
	a = np.nanmax([np.zeros(1036), np.arange(1036)-95], axis=0) 
	b = np.min([np.ones(1036)*1035, a+190], axis=0) 
	c = np.nanmin([np.ones(1036)*(1035-190), b-190], axis=0) 
	filter_min = np.array(c, dtype=int) 
	filter_max = np.array(b, dtype=int)


	for l_min, l_max in [(5455,5470),(5075,5095),(4355,4370)]: 
		wlhere = (def_wave >= l_min) & (def_wave <= l_max) 
		ffskysub[:,wlhere] = np.nan

	perc = 93 
	wlcont_lo = (def_wave > 4000)&(def_wave <= 4500) 
	medians_lo = np.nanmedian(ffskysub[:,wlcont_lo], axis=1) 
	perc_lo = np.nanpercentile(medians_lo, perc) 
	wlcont_hi = (def_wave > 4800)&(def_wave <= 5300) 
	medians_hi = np.nanmedian(ffskysub[:,wlcont_hi], axis=1) 
	perc_hi = np.nanpercentile(medians_hi, perc) 
	ffskysub[abs(medians_lo)>perc_lo] *= np.nan 
	ffskysub[abs(medians_hi)>perc_hi] *= np.nan

	residual_0 = np.nanmedian(ffskysub, axis=0)
	ffskysub = ffskysub - residual_0

	tab = {"wavelength": def_wave, "residual_0": residual_0}

	POLYCONTSUB = False

	if POLYCONTSUB:
		for kappa in [1, 3, 4, 5, 10]:
			continuum = []
			for i in range(len(ffskysub)):
				here = np.isfinite(ffskysub[i])
				std = biweight_scale(ffskysub[i][here])
				mid = biweight_location(ffskysub[i][here])
				here *= abs(ffskysub[i] - mid) < kappa*std
				if len(here[here]) < 1:
					continuum.append(np.nan * def_wave)
					continue
				ps = np.polyfit(def_wave[here], ffskysub[i][here], 5)
				continuum.append(ps[0] * def_wave**5 + ps[1] * def_wave**4 + ps[2] * def_wave**3 + ps[3] * def_wave**2 + ps[4] * def_wave + ps[5])
			continuum = np.array(continuum)
			continuum_subtracted = ffskysub.copy() - continuum
			continuum_subtracted[continuum_subtracted==0.0] = np.nan

			residual_continuum_subtracted = np.nanmedian(continuum_subtracted, axis=0)
			tab["residual_contsub_{}".format(kappa)] = residual_continuum_subtracted
	else:
		continuum = np.zeros(ffskysub.shape)
		indices = np.arange(1036)
		for i in indices:
			idxhere = (indices >= filter_min[i])&(indices <= filter_max[i])
			continuum[:,i] += np.nanmedian(ffskysub[:,idxhere], axis=1)
		continuum[continuum==0.0] = np.nan
		continuum_subtracted = ffskysub.copy() - continuum
		continuum_subtracted[continuum_subtracted==0.0] = np.nan
		tab["residual_contsub"] = np.nanmedian(continuum_subtracted, axis=0)

	ascii.write(tab, "residuals_continuum_subtraction_poly_{}.tab".format(shotid))
