import h5py
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)

import astropy.units as u
from astropy.convolution import convolve_fft
from astropy.io import ascii, fits

from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

def fit_moffat(dist, amp, fwhm):
	beta = 3.
	gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
	norm = (beta-1)/(np.pi*gamma**2)
	return amp * norm * (1+(dist/gamma)**2)**(-1*beta)

kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

max_radius = 50
diff = 0.1
tmp = np.arange(-max_radius, max_radius+diff, diff)
print(len(tmp))
INDEX = np.argmin(abs(tmp))
print("Middle row:", INDEX)
x = np.array([tmp for i in range(len(tmp))])
y = x.T
dist_map = np.sqrt(x**2+y**2)
dist_map_kpc = dist_map * kpc_per_arcsec_mid  # I pretend that I am observing the profiles at z=2.5
ones = np.ones(dist_map.shape)

amp=1
def get_convolved_profile_chris(fwhm):

	kernel = fit_moffat(dist_map, amp=1, fwhm=fwhm)
	lae_img = chris_profile_func(dist_map_kpc)
	result = convolve_fft(lae_img, kernel)
	
	kernel_fiber = np.where(dist_map <= 0.75, 1, 0)
	kernel_fiber_int = np.nansum(kernel_fiber)*diff**2
	kernel_fiber = kernel_fiber/kernel_fiber_int
	result_2 = convolve_fft(result, kernel_fiber)

	result_dict = {"r": dist_map[INDEX, INDEX:],
		 "rofile": result[INDEX, INDEX:],
		"profile_fiber": result_2[INDEX, INDEX:]}
																				
	return result_dict  

fn = "../karls_suggestion/"+"SBr_NIEMEYER_z3_veryextended.hdf5" # change filename to hdf5 I sent you

   
rbins = np.arange(0.5, 400.5, 1) # radial central bins in pkpc


for log_min, log_max in [(9.5, 10), (10, 10.5), (10.5, 11)]:
	sm_min = 10**log_min
	sm_max = 10**log_max

	stack_dict = {}
	with h5py.File(fn,"r") as hf:

		sm = hf["StellarMass30kpc"][:]
		bg = hf["background"].value

		mask = (sm>=sm_min) & (sm<sm_max) # in this example, let's restrict the stellar mass range we are interested in.
		for key in hf.keys():
			if key in ["r", "background", "StellarMass30kpc"]:
				continue
			tmp = hf[key][:]
			stack_dict[key] = np.nanmedian(tmp[:,mask], axis=1)
		
		print(hf.keys())

	convolved_dict = {}
	N = len(stack_dict.keys())
	i = 1
	for key in stack_dict.keys():
		stack = stack_dict[key]
		name = key
	
		chris_profile_func = interp1d(rbins, stack, bounds_error=False, fill_value=(stack[0], np.nan))
		result_dict = get_convolved_profile_chris(1.3)
		
		convolved_dict["r"] = result_dict["r"]
		convolved_dict[name] = result_dict["profile_fiber"]
		print(f"Finished {i}/{N}.")
		i += 1
		
	ascii.write(convolved_dict, "../karls_suggestion/chris_profiles/chris_profiles_extended_individual_{}_{}.tab".format(log_min, log_max))
	print("Finished mass range", log_min, log_max)
