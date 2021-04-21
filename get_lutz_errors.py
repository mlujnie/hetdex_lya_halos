# r/kpc delta_r/kpc median err_median median_contsub err_median_contsub weighted_biweight err_weighted_biweight weighted_biweight_contsub err_weighted_biweight_contsub median_lum1 median_lum2 median_lum3 err_median_lum1 err_median_lum2 err_median_lum3 median_broad err_median_broad median_narrow err_median_narrow

import glob
import numpy as np
from astropy.io import ascii
from astropy.stats import biweight_location, biweight_scale

lutz_errors = {}
for appendix in ["", "_4", "_11"]:
	# proper stacks
	median_stacks = []
	median_contsub_stacks = []
	biweight_stacks = []
	biweight_contsub_stacks = []
	for fin in glob.glob("radial_profiles_empty_wlshift/*_proper_skymask_*.tab"):
		tmp = ascii.read(fin)
		median_stacks.append(tmp["median"+appendix])
		median_contsub_stacks.append(tmp["median_contsub"+appendix])
		biweight_stacks.append(tmp["weighted_biweight"+appendix])
		biweight_contsub_stacks.append(tmp["weighted_biweight_contsub"+appendix])

	std_median_stacks = np.nanstd(median_stacks, axis=0)
	std_median_contsub_stacks = np.nanstd(median_contsub_stacks, axis=0)
	std_wb_stacks = np.nanstd(biweight_stacks, axis=0)
	std_wb_contsub_stacks = np.nanstd(biweight_contsub_stacks, axis=0)

	lutz_errors["r/kpc"] = tmp["r/kpc"]
	lutz_errors["std_median_stacks"+appendix] = std_median_stacks
	lutz_errors["std_median_contsub_stacks"+appendix] = std_median_contsub_stacks
	lutz_errors["std_wb_stacks"+appendix] = std_wb_stacks
	lutz_errors["std_wb_contsub_stacks"+appendix] = std_wb_contsub_stacks

ascii.write(lutz_errors, "stack_errors_empirical_muse_skymask.tab")
print("Wrote errors to stack_errors_empirical_muse_skymask.tab")
