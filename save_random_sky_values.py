import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9
import astropy.units as u
import tables as tb
import sys
from scipy.interpolate import interpn
from astropy.convolution import convolve_fft
import time

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
# update matplotlib params for bigger font
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral'}
pylab.rcParams.update(params)

import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation, biweight_scale

# read in interpolation grid
from scipy.interpolate import RegularGridInterpolator
INTERPOLATOR = RegularGridInterpolator

from scipy.ndimage import gaussian_filter
import glob
import pickle

# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147) 

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]

# some functions
import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation


def biweight_location_weights(data, weights, c=6.0, M=None, axis=None):
    """ weighted biweight location a la Karl
        nan-resistant and excludes data with zero weights"""

    data = np.asanyarray(data).astype(np.float64)
    weights = np.asanyarray(weights).astype(np.float64)
   
    data[weights==0] = np.nan 
    weights[~np.isfinite(data)] = np.nan
    
    if (data.shape!=weights.shape):
        raise ValueError("data.shape != weights.shape")

    if M is None:
        M = np.nanmedian(data, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
    #madweights = median_absolute_deviation(weights, axis=axis)

    if axis is None and mad == 0.:
        return M  # return median if data is a constant array
    
    #if axis is None and madweights == 0:
    #    madweights = 1.

    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
        const_mask = (mad == 0.)
        mad[const_mask] = 1.  # prevent divide by zero

    #if axis is not None:
    #    madweights = np.expand_dims(madweights, axis=axis)
    #    const_mask = (madweights == 0.)
    #    madweights[const_mask] = 1.  # prevent divide by zero

    cmadsq = (c*mad)**2
    
    factor = 0.5
    weights  = weights/np.nanmedian(weights)*factor 
    
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    #print("number of excluded points ", len(mask[mask]))
    
    u = (1 - u ** 2) ** 2
    
    weights[~np.isfinite(weights)] = 0
    
    u = u + weights**2
    u[weights==0] = 0
    d[weights==0] = 0
    u[mask] = 0

    # along the input axis if data is constant, d will be zero, thus
    # the median value will be returned along that axis
    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)



# new bogus LAEs
print("Reading bogus LAEs...")
bogus_list = []
i=0
N = len(sources)
for source in sources:
    detectid = source["detectid"]
    lae_file = "../radial_profiles/bogus_laes_skymask/bogus_0_{}.dat".format(detectid)
    try:
        lae_file = glob.glob(lae_file)[0]
        lae_tab = ascii.read(lae_file)
        lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
        lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
        bogus_list.append(lae_tab)
        i+=1
        if i%100 == 0:
            print(f"Finished {i}/{N}")
    except Exception as e:
        print("Failed to read "+lae_file)

    lae_file = "../radial_profiles/bogus_laes_skymask/bogus_1_{}.dat".format(detectid)
    try:
        lae_file = glob.glob(lae_file)[0]
        lae_tab = ascii.read(lae_file)
        lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
        lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
        bogus_list.append(lae_tab)
        i+=1
        if i%100 == 0:
            print(f"Finished {i}/{N}")
    except Exception as e:
        print("Failed to read "+lae_file)


bogus_tab = vstack(bogus_list)
bogus_tab["mask_7"] = bogus_tab["mask_7"].astype(bool)
bogus_tab["mask_10"] = bogus_tab["mask_10"].astype(bool)
print("See if masking works: ", len(bogus_tab), len(bogus_tab[bogus_tab["mask_7"]]))



# bogus values
bogus_dict = {}
fiberarea = np.pi*(0.75)**2

total_randoms = bogus_tab["flux"][bogus_tab["mask_7"]] / fiberarea
total_randoms_contsub = bogus_tab["flux_contsub"][bogus_tab["mask_7"]] / fiberarea
total_weights = bogus_tab["sigma"][bogus_tab["mask_7"]]**(-2)

fig = plt.figure(figsize=[6.4, 4.8])
mid = np.nanmedian(total_randoms_contsub)
std = np.nanstd(total_randoms_contsub)
bstd = biweight_scale(total_randoms_contsub)

bins= np.arange(-2.5, 2.5, 0.02)
plt.hist(bogus_tab["flux_contsub"]/fiberarea, bins=bins, log=False, density=False, alpha=0.5, label="all")
plt.hist(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiberarea, bins=bins, log=False, density=False, alpha=0.5, label="cleaned")
plt.axvline(mid, color="red", labeL="median")
plt.xlabel(r"surface brightness [$10^{-17} \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{arcsec}^{-2}$]")
plt.ylabel("N")
plt.axvline(mid+std, linestyle=":", color="red")
plt.axvline(mid-std, linestyle=":", color="red", label="std")
plt.axvline(mid+bstd, linestyle="--", color="red")
plt.axvline(mid-bstd, linestyle="--", color="red", label="biweight scale")
plt.axvline(np.nanpercentile(total_randoms_contsub, 16), linestyle="--", color="blue", label="16/84 percentile")
plt.axvline(np.nanpercentile(total_randoms_contsub, 100-16), linestyle="--", color="blue")

plt.legend(bbox_to_anchor=(1,1))
fig.set_facecolor("white")
fig.savefig("bogus_distribution_flux_contsub_final.pdf", bbox_inches="tight")

random_value = np.nanmedian(total_randoms)
random_value_contsub = np.nanmedian(total_randoms_contsub)
random_value_contsub_mean = np.nanmean(total_randoms_contsub)
mask = np.isfinite(total_randoms) & np.isfinite(total_weights) & (total_weights < 7.5) & np.isfinite(total_randoms_contsub)
random_value_bw = biweight_location_weights(total_randoms[mask], weights=total_weights[mask])
random_value_contsub_bw = biweight_location_weights(total_randoms_contsub[mask], weights=total_weights[mask])

perc_16 = np.nanpercentile(total_randoms, 16)
perc_84 = np.nanpercentile(total_randoms, 84)
err_median_perc_lower = (random_value - perc_16)/np.sqrt(len(total_randoms))
err_median_perc_upper = (perc_84 - random_value)/np.sqrt(len(total_randoms))

perc_16_contsub = np.nanpercentile(total_randoms_contsub, 16)
perc_84_contsub = np.nanpercentile(total_randoms_contsub, 84)
err_median_perc_lower_contsub = (random_value_contsub - perc_16_contsub)/np.sqrt(len(total_randoms_contsub))
err_median_perc_upper_contsub = (perc_84_contsub - random_value_contsub)/np.sqrt(len(total_randoms_contsub))

random_value_error = np.nanstd(total_randoms)/np.sqrt(len(total_randoms))
random_value_contsub_error = np.nanstd(total_randoms_contsub)/np.sqrt(len(total_randoms_contsub))
random_value_bw_error = biweight_scale(total_randoms[mask])/np.sqrt(len(total_randoms[mask]))
random_value_contsub_bw_error = biweight_scale(total_randoms_contsub[mask])/np.sqrt(len(total_randoms_contsub[mask]))

bogus_dict["median"] = [random_value]
bogus_dict["err_median"] = [random_value_error]
bogus_dict["err_median_perc_lower"] = [err_median_perc_lower]
bogus_dict["err_median_perc_upper"] = [err_median_perc_upper]
bogus_dict["median_contsub"] = [random_value_contsub]
bogus_dict["mean_contsub"] = [random_value_contsub_mean]
bogus_dict["err_median_contsub"] = [random_value_contsub_error]
bogus_dict["err_median_perc_lower_contsub"] = [err_median_perc_lower_contsub]
bogus_dict["err_median_perc_upper_contsub"] = [err_median_perc_upper_contsub]
bogus_dict["weighted_biweight"] = [random_value_bw]
bogus_dict["err_weighted_biweight"] = [random_value_bw_error]
bogus_dict["weighted_biweight_contsub"] = [random_value_contsub_bw]
bogus_dict["err_weighted_biweight_contsub"] = [random_value_contsub_bw_error]

ascii.write(bogus_dict, "random_sky_values_skymask_double.tab")
print("Wrote to random_sky_values_skymask_double.tab")

# bogus values: fixed line width 4AA
bogus_dict = {}
fiberarea = np.pi*(0.75)**2

total_randoms = bogus_tab["flux_4"][bogus_tab["mask_7"]] / fiberarea
total_randoms_contsub = bogus_tab["flux_contsub_4"][bogus_tab["mask_7"]] / fiberarea
total_weights = bogus_tab["sigma_4"][bogus_tab["mask_7"]]**(-2)

random_value = np.nanmedian(total_randoms)
random_value_contsub = np.nanmedian(total_randoms_contsub)
mask = np.isfinite(total_randoms) & np.isfinite(total_weights) & (total_weights < 7.5) & np.isfinite(total_randoms_contsub)
random_value_bw = biweight_location_weights(total_randoms[mask], weights=total_weights[mask])
random_value_contsub_bw = biweight_location_weights(total_randoms_contsub[mask], weights=total_weights[mask])

perc_16 = np.nanpercentile(total_randoms, 16)
perc_84 = np.nanpercentile(total_randoms, 84)
err_median_perc_lower = (random_value - perc_16)/np.sqrt(len(total_randoms))
err_median_perc_upper = (perc_84 - random_value)/np.sqrt(len(total_randoms))

perc_16_contsub = np.nanpercentile(total_randoms_contsub, 16)
perc_84_contsub = np.nanpercentile(total_randoms_contsub, 84)
err_median_perc_lower_contsub = (random_value_contsub - perc_16_contsub)/np.sqrt(len(total_randoms_contsub))
err_median_perc_upper_contsub = (perc_84_contsub - random_value_contsub)/np.sqrt(len(total_randoms_contsub))

random_value_error = np.nanstd(total_randoms)/np.sqrt(len(total_randoms))
random_value_contsub_error = np.nanstd(total_randoms_contsub)/np.sqrt(len(total_randoms_contsub))
random_value_bw_error = biweight_scale(total_randoms[mask])/np.sqrt(len(total_randoms[mask]))
random_value_contsub_bw_error = biweight_scale(total_randoms_contsub[mask])/np.sqrt(len(total_randoms_contsub[mask]))

bogus_dict["median"] = [random_value]
bogus_dict["err_median"] = [random_value_error]
bogus_dict["err_median_perc_lower"] = [err_median_perc_lower]
bogus_dict["err_median_perc_upper"] = [err_median_perc_upper]
bogus_dict["median_contsub"] = [random_value_contsub]
bogus_dict["err_median_contsub"] = [random_value_contsub_error]
bogus_dict["err_median_perc_lower_contsub"] = [err_median_perc_lower_contsub]
bogus_dict["err_median_perc_upper_contsub"] = [err_median_perc_upper_contsub]
bogus_dict["weighted_biweight"] = [random_value_bw]
bogus_dict["err_weighted_biweight"] = [random_value_bw_error]
bogus_dict["weighted_biweight_contsub"] = [random_value_contsub_bw]
bogus_dict["err_weighted_biweight_contsub"] = [random_value_contsub_bw_error]

ascii.write(bogus_dict, "random_sky_values_skymask_double_4.tab")
print("Wrote to random_sky_values_skymask_double_4.tab")


# bogus values: fixed line width 11AA
bogus_dict = {}
fiberarea = np.pi*(0.75)**2

total_randoms = bogus_tab["flux_11"][bogus_tab["mask_7"]] / fiberarea
total_randoms_contsub = bogus_tab["flux_contsub_11"][bogus_tab["mask_7"]] / fiberarea
total_weights = bogus_tab["sigma_11"][bogus_tab["mask_7"]]**(-2)

random_value = np.nanmedian(total_randoms)
random_value_contsub = np.nanmedian(total_randoms_contsub)
mask = np.isfinite(total_randoms) & np.isfinite(total_weights) & (total_weights < 7.5) & np.isfinite(total_randoms_contsub)
random_value_bw = biweight_location_weights(total_randoms[mask], weights=total_weights[mask])
random_value_contsub_bw = biweight_location_weights(total_randoms_contsub[mask], weights=total_weights[mask])

perc_16 = np.nanpercentile(total_randoms, 16)
perc_84 = np.nanpercentile(total_randoms, 84)
err_median_perc_lower = (random_value - perc_16)/np.sqrt(len(total_randoms))
err_median_perc_upper = (perc_84 - random_value)/np.sqrt(len(total_randoms))

perc_16_contsub = np.nanpercentile(total_randoms_contsub, 16)
perc_84_contsub = np.nanpercentile(total_randoms_contsub, 84)
err_median_perc_lower_contsub = (random_value_contsub - perc_16_contsub)/np.sqrt(len(total_randoms_contsub))
err_median_perc_upper_contsub = (perc_84_contsub - random_value_contsub)/np.sqrt(len(total_randoms_contsub))

random_value_error = np.nanstd(total_randoms)/np.sqrt(len(total_randoms))
random_value_contsub_error = np.nanstd(total_randoms_contsub)/np.sqrt(len(total_randoms_contsub))
random_value_bw_error = biweight_scale(total_randoms[mask])/np.sqrt(len(total_randoms[mask]))
random_value_contsub_bw_error = biweight_scale(total_randoms_contsub[mask])/np.sqrt(len(total_randoms_contsub[mask]))

bogus_dict["median"] = [random_value]
bogus_dict["err_median"] = [random_value_error]
bogus_dict["err_median_perc_lower"] = [err_median_perc_lower]
bogus_dict["err_median_perc_upper"] = [err_median_perc_upper]
bogus_dict["median_contsub"] = [random_value_contsub]
bogus_dict["err_median_contsub"] = [random_value_contsub_error]
bogus_dict["err_median_perc_lower_contsub"] = [err_median_perc_lower_contsub]
bogus_dict["err_median_perc_upper_contsub"] = [err_median_perc_upper_contsub]
bogus_dict["weighted_biweight"] = [random_value_bw]
bogus_dict["err_weighted_biweight"] = [random_value_bw_error]
bogus_dict["weighted_biweight_contsub"] = [random_value_contsub_bw]
bogus_dict["err_weighted_biweight_contsub"] = [random_value_contsub_bw_error]

ascii.write(bogus_dict, "random_sky_values_skymask_double_11.tab")
print("Wrote to random_sky_values_skymask_double_11.tab")
