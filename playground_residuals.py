import astropy.units as u 
from astropy.coordinates import SkyCoord
from hetdex_api.shot import *
import numpy as np
from astropy.io import ascii

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
print(len(sources))
shot_ids = np.unique(sources["shotid"])

for shotid in [20190105012]:
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

	continuum = np.zeros(ffskysub.shape) 
	indices = np.arange(1036) 
	for i in indices: 
		idxhere = (indices >= filter_min[i])&(indices <= filter_max[i]) 
		continuum[:,i] += np.nanmedian(ffskysub[:,idxhere], axis=1)

	continuum[continuum==0] = np.nan
	continuum_subtracted = ffskysub.copy() - continuum
	continuum_subtracted[continuum_subtracted==0.0] = np.nan

	residual = np.nanmedian(ffskysub, axis=0)
	residual_continuum_subtracted = np.nanmedian(continuum_subtracted, axis=0)

	tab = {"wavelength": def_wave, "residual": residual, "residual_contsub": residual_continuum_subtracted}
	ascii.write(tab, "residuals_continuum_subtracted_vs_not_{}.tab".format(shotid))
