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

def lae_powerlaw_profile(r, c1, c2):
    return c1*psf_func((FWHM, r)) + c2*powerlaw_func((FWHM, r))

def fit_moffat(dist, amp, fwhm):
	beta = 3.
	gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
	norm = (beta-1)/(np.pi*gamma**2)
	return amp * norm * (1+(dist/gamma)**2)**(-1*beta)

# read in stars
star_list = []
star_sources = ascii.read("star_gaia_tab.tab")
i=0
N = 3795
#for source in star_sources:
#    detectid = source["detectid"]
#    lae_file = "../radial_profiles/stars_skymask/star_{}.dat".format(detectid)
for lae_file in glob.glob("../radial_profiles/stars_skymask/star_*.dat"):
    try:
        #lae_file = glob.glob(lae_file)[0]
        lae_tab = ascii.read(lae_file)
        lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
        lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
        star_list.append(lae_tab)
        i+=1
        if i%100 == 0:
            print(f"Finished {i}/{N}")
    except Exception as e:
        #print("Failed to read "+lae_file)
        print(e)
        pass

        
star_tab= vstack(star_list)
star_tab["mask_7"] = star_tab["mask_7"].astype(bool)
star_tab["mask_10"] = star_tab["mask_10"].astype(bool)

star_tab = star_tab[np.isfinite(star_tab["flux"])&np.isfinite(star_tab["sigma"])]


# stack stars
r_bins = [ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5, 5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]

stack_r, stack_flux, stack_error, stack_flux_bw = [], [], [], []
for r_min, r_max in zip(r_bins[:-1], r_bins[1:]):
    stack_r.append(r_min+(r_max-r_min)/2.)
    stack_flux.append(np.nanmedian(star_tab["flux"][(star_tab["r"]>=r_min)&(star_tab["r"]<r_max)]))
    stack_flux_bw.append(biweight_location(star_tab["flux"][(star_tab["r"]>=r_min)&(star_tab["r"]<r_max)]))
    stack_error.append(biweight_scale(star_tab["flux"][(star_tab["r"]>=r_min)&(star_tab["r"]<r_max)])/np.sqrt(len(star_tab["flux"][(star_tab["r"]>=r_min)&(star_tab["r"]<r_max)])))

stack_r = np.array(stack_r)
stack_flux = np.array(stack_flux)
stack_flux_bw = np.array(stack_flux_bw)
stack_error = np.array(stack_error)

stack_integral = np.nansum(stack_flux[stack_r<=2.0])*(stack_r[1]-stack_r[0])
stack_integral_bw = np.nansum(stack_flux_bw[stack_r<=2.0])*(stack_r[1]-stack_r[0])

rad_tab = {"r": stack_r, "median": stack_flux, "biweight_location": stack_flux_bw, "bw_error": stack_error}

ascii.write(rad_tab, "star_stack_radprof.tab")
print("Wrote to stack_star_radprof.tab")


# save median/biweight location of unmasked regions in this area to get a random value

bogus_tab = star_tab[star_tab["mask_7"]]
print("len(bogus_tab): ", len(bogus_tab))
total_randoms = bogus_tab["flux"]
total_weights = bogus_tab["sigma"]**(-2)

random_value = np.nanmedian(total_randoms)
mask = np.isfinite(total_randoms) & np.isfinite(total_weights) & (total_weights < 7.5)
random_value_bw = biweight_location_weights(total_randoms[mask], weights=total_weights[mask])
random_value_biweight = biweight_location(total_randoms[mask])

perc_16 = np.nanpercentile(total_randoms, 16)
perc_84 = np.nanpercentile(total_randoms, 84)
err_median_perc_lower = (random_value - perc_16)/np.sqrt(len(total_randoms))
err_biweight_perc_lower = (random_value - perc_16)/np.sqrt(len(total_randoms))
err_median_perc_upper = (perc_84 - random_value_biweight)/np.sqrt(len(total_randoms))
err_biweight_perc_upper = (perc_84 - random_value_biweight)/np.sqrt(len(total_randoms))

random_value_error = np.nanstd(total_randoms)/np.sqrt(len(total_randoms))
random_value_bw_error = biweight_scale(total_randoms[mask])/np.sqrt(len(total_randoms[mask]))

bogus_dict = {}
bogus_dict["median"] = [random_value]
bogus_dict["err_median"] = [random_value_error]
bogus_dict["err_median_perc_lower"] = [err_median_perc_lower]
bogus_dict["err_biweight_perc_lower"] = [err_biweight_perc_lower]
bogus_dict["err_median_perc_upper"] = [err_median_perc_upper]
bogus_dict["err_biweight_perc_upper"] = [err_biweight_perc_upper]
bogus_dict["weighted_biweight"] = [random_value_bw]
bogus_dict["err_biweight"] = [random_value_bw_error]
bogus_dict["biweight"] = [random_value_biweight]
bogus_dict["stack_integral_median"] = [stack_integral]
bogus_dict["stack_integral_biweight"] = [stack_integral_bw]

ascii.write(bogus_dict, "empty_sky_values_stars.tab")
print("Wrote to empty_sky_values_stars.tab")


