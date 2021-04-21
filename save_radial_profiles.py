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

def get_stack(r_bins_min, r_bins_max, data_r, data_flux, kind="median", weights_flux=None):
    
    stack, e_stack = [], []
    for r_min, r_max in zip(r_bins_min, r_bins_max):
        here = (data_r < r_max)&(data_r >= r_min)
        tmp_flux = data_flux[here]
        if weights_flux is not None:
            tmp_weights = weights_flux[here]
        if kind == "median":
            stack.append(np.nanmedian(tmp_flux))
            e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
        elif kind == "mean":
            stack.append(np.nanmean(tmp_flux))
            e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
        elif kind == "biweight":
            stack.append(biweight_location(tmp_flux, ignore_nan=True))
            e_stack.append(biweight_scale(tmp_flux, ignore_nan=True)/np.sqrt(len(tmp_flux)))
        elif kind == "weighted_biweight":
            if weights_flux is None:
                print("You need to provide weights!")
                return 0, 0, 0
            mask = np.isfinite(tmp_flux)&(np.isfinite(tmp_weights))
            stack.append(biweight_location_weights(tmp_flux[mask], tmp_weights[mask]))
            e_stack.append(biweight_scale(tmp_flux[mask])/np.sqrt(len(tmp_flux[mask])))
        else:
            print("Unknown kind: ", kind)
            return 0, 0, 0
    r_bins_mid = np.nanmean([r_bins_min, r_bins_max], axis=0)
    fiberarea = np.pi*0.75**2
    return r_bins_mid, np.array(stack)/fiberarea, np.array(e_stack)/fiberarea

def get_stack_proper(r_bins_min, r_bins_max, data_r, data_flux, data_redshift, kind="median", weights_flux=None):
    
    stack, e_stack = [], []
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(data_redshift)/60*u.arcmin/u.kpc
    for r_min, r_max in zip(r_bins_min, r_bins_max):
        here = (data_r*kpc_per_arcsec < r_max)&(data_r*kpc_per_arcsec >= r_min)
        tmp_flux = data_flux[here]
        if weights_flux is not None:
            tmp_weights = weights_flux[here]
        if kind == "median":
            stack.append(np.nanmedian(tmp_flux))
            e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
        elif kind == "mean":
            stack.append(np.nanmean(tmp_flux))
            e_stack.append(np.nanstd(tmp_flux)/np.sqrt(len(tmp_flux)))
        elif kind == "biweight":
            stack.append(biweight_location(tmp_flux, ignore_nan=True))
            e_stack.append(biweight_scale(tmp_flux, ignore_nan=True)/np.sqrt(len(tmp_flux)))
        elif kind == "weighted_biweight":
            if weights_flux is None:
                print("You need to provide weights!")
                return 0, 0, 0
            mask = np.isfinite(tmp_flux)&(np.isfinite(tmp_weights))
            stack.append(biweight_location_weights(tmp_flux[mask], tmp_weights[mask]))
            e_stack.append(biweight_scale(tmp_flux[mask])/np.sqrt(len(tmp_flux[mask])))
        else:
            print("Unknown kind: ", kind)
            return 0, 0, 0
    r_bins_mid = np.nanmean([r_bins_min, r_bins_max], axis=0)
    fiberarea = np.pi*0.75**2
    return r_bins_mid, np.array(stack)/fiberarea, np.array(e_stack)/fiberarea

# read in data
sources = ascii.read("../karls_suggestion/high_sn_sources.tab")
sources["mask"] = sources["mask"].astype(bool)
sources = sources[sources["mask"]]
print(len(sources), " remaining.")

# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
luminositites = sources["flux"]*cosmo.luminosity_distance(z_lae)**2*4*np.pi/u.Mpc**2
sources["luminosity"] = luminositites

# real LAE data
print("Reading LAEs...")
long_list = []
i=0
N = len(sources)
mask = []
for source in sources:
    detectid = source["detectid"]
    wavelength = source["wave"]
    redshift = wavelength/1215.67 - 1
    lae_file = "../radial_profiles/laes_skymask/lae_{}.dat".format(detectid)
    lae_file = glob.glob(lae_file)[0]
    try:
        lae_tab = ascii.read(lae_file)
        if len(lae_tab) < 1:
            continue
        lae_tab["detectid"] = [detectid for j in range(len(lae_tab))]
        lae_tab["redshift"] = [redshift for j in range(len(lae_tab))]
        lae_tab["mask_7"] = lae_tab["mask_7"].astype(str)
        long_list.append(lae_tab)
        mask.append(source["linewidth"] < 7.)
        i+=1
        if i%100 == 0:
            print(f"Finished {i}/{N}")
    except Exception as e:
        print("Failed to read "+lae_file)
print("{} LAEs are non-empty.".format(i))
        
long_tab = vstack(long_list)
long_tab["mask_7"] = (long_tab["mask_7"]=="True").astype(bool)
long_tab["mask_10"] = (long_tab["mask_10"]=="True").astype(bool)
print("See if masking works: ", len(long_tab), len(long_tab[long_tab["mask_7"]]))

# Lutz profile
lutz = ascii.read("../jupyter/lutz_profile.tab")

# getting the stacks
kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

r_bins_kpc = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 160])
r_bins_max_kpc = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 160, 320])
r_bins_plot_kpc = np.nanmean([r_bins_kpc, r_bins_max_kpc], axis=0)
r_bins_kpc_xbars = (r_bins_max_kpc-r_bins_kpc)/2.

r_bins = r_bins_kpc / kpc_per_arcsec_mid
r_bins_max = r_bins_max_kpc / kpc_per_arcsec_mid
r_bins_plot = r_bins_plot_kpc / kpc_per_arcsec_mid 
r_bins_xbars = (r_bins_max-r_bins)/2.

big_tab_angular = {}
big_tab_proper = {}

# LAEs median stack
here = long_tab["mask_7"]
r, sb_median, e_sb_median = get_stack(r_bins, r_bins_max, long_tab["r"][here], long_tab["flux"][here])

r, sb_median_contsub, e_sb_median_contsub = get_stack(r_bins, r_bins_max, long_tab["r"][here], long_tab["flux_contsub"][here])

big_tab_angular["r"] = r
big_tab_angular["delta_r"] = r_bins_xbars
big_tab_angular["median"] = sb_median
big_tab_angular["err_median"] = e_sb_median
big_tab_angular["median_contsub"] = sb_median_contsub
big_tab_angular["err_median_contsub"] = e_sb_median_contsub

ascii.write(big_tab_angular, "radial_profiles_angular_skymask.tab")
print("Wrote to radial_profiles_angular_nb.tab")

# LAEs proper (median)
here = long_tab["mask_7"]
r_kpc, sb_median_kpc, e_sb_median_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux"][here], long_tab["redshift"][here])

r_kpc, sb_median_contsub_kpc, e_sb_median_contsub_kpc = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub"][here], long_tab["redshift"][here])

# LAEs proper (median) fixed linewidth 4AA
here = long_tab["mask_7"]
r_kpc, sb_median_kpc_4, e_sb_median_kpc_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_4"][here], long_tab["redshift"][here])

r_kpc, sb_median_contsub_kpc_4, e_sb_median_contsub_kpc_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_4"][here], long_tab["redshift"][here])

# LAEs proper (median) fixed linewidth 11AA
here = long_tab["mask_7"]
r_kpc, sb_median_kpc_11, e_sb_median_kpc_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_11"][here], long_tab["redshift"][here])

r_kpc, sb_median_contsub_kpc_11, e_sb_median_contsub_kpc_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_11"][here], long_tab["redshift"][here])

# LAEs proper: weighted biweight
here = long_tab["mask_7"]
weights = long_tab["sigma"]**(-2)
here *= weights < 7.5
r_kpc, sb_median_kpc_wb, e_sb_median_kpc_wb = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux"][here], long_tab["redshift"][here], kind="weighted_biweight", weights_flux=weights[here])

r_kpc, sb_median_contsub_kpc_wb, e_sb_median_contsub_kpc_wb = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub"][here], long_tab["redshift"][here], kind="weighted_biweight", weights_flux=weights[here])

# broad line sources
linewidth_mask = np.ones(len(long_tab), dtype=bool)
for source in sources:
    linewidth_mask[long_tab["detectid"]==source["detectid"]] = source["linewidth"] > 7.
print("Number of sources with linewidth > 7AA: ", len(sources[sources["linewidth"]>7.]))
print("Number of sources with linewidth <= 7AA: ", len(sources[sources["linewidth"]<=7.]))

# variable line width
here = long_tab["mask_7"] & linewidth_mask
print(len(here[here]), len(here))
r, sb_median_broad, e_sb_median_broad = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux"][here], long_tab["redshift"][here])

r, sb_median_contsub_broad, e_sb_median_contsub_broad = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub"][here], long_tab["redshift"][here])

here = long_tab["mask_7"] & (~linewidth_mask)
print(len(here[here]), len(here))
r, sb_median_narrow, e_sb_median_narrow = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux"][here], long_tab["redshift"][here])

r, sb_median_contsub_narrow, e_sb_median_contsub_narrow = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub"][here], long_tab["redshift"][here])


# fixed line width 4AA
here = long_tab["mask_7"] & linewidth_mask
print(len(here[here]), len(here))
r, sb_median_broad_4, e_sb_median_broad_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_4"][here], long_tab["redshift"][here])

r, sb_median_contsub_broad_4, e_sb_median_contsub_broad_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_4"][here], long_tab["redshift"][here])

here = long_tab["mask_7"] & (~linewidth_mask)
print(len(here[here]), len(here))
r, sb_median_narrow_4, e_sb_median_narrow_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_4"][here], long_tab["redshift"][here])

r, sb_median_contsub_narrow_4, e_sb_median_contsub_narrow_4 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_4"][here], long_tab["redshift"][here])


# fixed line width 11AA
here = long_tab["mask_7"] & linewidth_mask
print(len(here[here]), len(here))
r, sb_median_broad_11, e_sb_median_broad_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_11"][here], long_tab["redshift"][here])

r, sb_median_contsub_broad_11, e_sb_median_contsub_broad_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_11"][here], long_tab["redshift"][here])

here = long_tab["mask_7"] & (~linewidth_mask)
print(len(here[here]), len(here))
r, sb_median_narrow_11, e_sb_median_narrow_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_11"][here], long_tab["redshift"][here])

r, sb_median_contsub_narrow_11, e_sb_median_contsub_narrow_11 = get_stack_proper(r_bins_kpc, r_bins_max_kpc, long_tab["r"][here], long_tab["flux_contsub_11"][here], long_tab["redshift"][here])

big_tab_proper["r/kpc"] = r_kpc
big_tab_proper["delta_r/kpc"] = r_bins_kpc_xbars
big_tab_proper["median"] = sb_median_kpc
big_tab_proper["median_4"] = sb_median_kpc_4
big_tab_proper["median_11"] = sb_median_kpc_11
big_tab_proper["err_median"] = e_sb_median_kpc
big_tab_proper["err_median_4"] = e_sb_median_kpc_4
big_tab_proper["err_median_11"] = e_sb_median_kpc_11
big_tab_proper["median_contsub"] = sb_median_contsub_kpc
big_tab_proper["median_contsub_4"] = sb_median_contsub_kpc_4
big_tab_proper["median_contsub_11"] = sb_median_contsub_kpc_11
big_tab_proper["err_median_contsub"] = e_sb_median_contsub_kpc
big_tab_proper["err_median_contsub_4"] = e_sb_median_contsub_kpc_4
big_tab_proper["err_median_contsub_11"] = e_sb_median_contsub_kpc_11
big_tab_proper["weighted_biweight"] = sb_median_kpc_wb
big_tab_proper["err_weighted_biweight"] = e_sb_median_kpc_wb
big_tab_proper["weighted_biweight_contsub"] = sb_median_contsub_kpc_wb
big_tab_proper["err_weighted_biweight_contsub"] = e_sb_median_contsub_kpc_wb
#big_tab_proper["median_lum1"] = sb_median_lum1
#big_tab_proper["median_lum2"] = sb_median_lum2
#big_tab_proper["median_lum3"] = sb_median_lum3
#big_tab_proper["err_median_lum1"] = e_sb_median_lum1
#big_tab_proper["err_median_lum2"] = e_sb_median_lum2
#big_tab_proper["err_median_lum3"] = e_sb_median_lum3
big_tab_proper["median_broad"] = sb_median_broad
big_tab_proper["median_broad_4"] = sb_median_broad_4
big_tab_proper["median_broad_11"] = sb_median_broad_11
big_tab_proper["err_median_broad"] = e_sb_median_broad
big_tab_proper["err_median_broad_4"] = e_sb_median_broad_4
big_tab_proper["err_median_broad_11"] = e_sb_median_broad_11
big_tab_proper["median_narrow"] = sb_median_narrow
big_tab_proper["median_narrow_4"] = sb_median_narrow_4
big_tab_proper["median_narrow_11"] = sb_median_narrow_11
big_tab_proper["err_median_narrow"] = e_sb_median_narrow
big_tab_proper["err_median_narrow_4"] = e_sb_median_narrow_4
big_tab_proper["err_median_narrow_11"] = e_sb_median_narrow_11

ascii.write(big_tab_proper, "radial_profiles_proper_skymask.tab")
print("Wrote to radial_profiles_proper_skymask.tab")


