# r/kpc delta_r/kpc median err_median median_contsub err_median_contsub weighted_biweight err_weighted_biweight weighted_biweight_contsub err_weighted_biweight_contsub median_lum1 median_lum2 median_lum3 err_median_lum1 err_median_lum2 err_median_lum3 median_broad err_median_broad median_narrow err_median_narrow

import glob
import numpy as np
from astropy.io import ascii
from astropy.stats import biweight_location, biweight_scale

lutz_errors = {}
stack_lists = {}
# proper stacks

SN_65 = True
if SN_65:
	ff = glob.glob("radial_profiles_empty_wlshift/*_proper_multimask_sn_65_*_unmasked.tab")
else:
	ff =  glob.glob("radial_profiles_empty_wlshift/*_proper_multimask_[!s]*.tab")
tmp = ascii.read(ff[0])

for key in tmp.keys():
	if key == "r/kpc":
		lutz_errors[key] = tmp[key]
		pass
	elif key == "delta_r/kpc":
		lutz_errors[key] = tmp[key]
		pass
	elif key[:3] == "err":
		pass
	else:
		stack_lists[key] = []

i = 0
for fin in ff:
	i+=1
	tmp = ascii.read(fin)

	for key in stack_lists.keys():
		stack_lists[key].append(tmp[key])
print("number of files used: ", i)
for key in stack_lists.keys():
	lutz_errors[key] = np.nanstd(stack_lists[key], axis=0)

if SN_65:
	filename = "stack_errors_empirical_muse_multimask_sn_65_unmasked.tab"
	ascii.write(lutz_errors, filename)
	print("Wrote to ", filename)
else:
	ascii.write(lutz_errors, "stack_errors_empirical_muse_multimask.tab")
	print("Wrote errors to stack_errors_empirical_muse_multimask.tab")
