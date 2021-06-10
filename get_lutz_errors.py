# r/kpc delta_r/kpc median err_median median_contsub err_median_contsub weighted_biweight err_weighted_biweight weighted_biweight_contsub err_weighted_biweight_contsub median_lum1 median_lum2 median_lum3 err_median_lum1 err_median_lum2 err_median_lum3 median_broad err_median_broad median_narrow err_median_narrow

import glob
import numpy as np
from astropy.io import ascii
from astropy.stats import biweight_location, biweight_scale
import argparse
import os
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--final_dir', type=str, default=".", help='Directory to save radial profiles. This is necessary to add.')
args = parser.parse_args(sys.argv[1:])

fmtstr = " Name: %(user_name)s : %(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s"
datestr = "%m/%d/%Y %I:%M:%S %p "
#basic logging config
logging.basicConfig(
        filename=os.path.join(args.final_dir, "get_lutz_errors.log"),
        level=logging.DEBUG,
        filemode="w",
        datefmt=datestr,
)

logging.info('Final directory: {}'.format(args.final_dir))

lutz_errors = {}
stack_lists = {}
# proper stacks

ff = glob.glob(os.path.join(args.final_dir, "radial_profiles_empty_wlshift/radial_profiles_proper_multimask*.tab"))
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
logging.info("number of files used: {}".format( i))
for key in stack_lists.keys():
	lutz_errors[key] = np.nanstd(stack_lists[key], axis=0)

filename = os.path.join(args.final_dir, 'stack_errors_empirical_muse_multimask.tab')
ascii.write(lutz_errors, filename)
logging.info("Wrote errors to {}".format(filename))
