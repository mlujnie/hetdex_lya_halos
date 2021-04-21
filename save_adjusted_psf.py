from astropy.modeling.functional_models import Sersic2D
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
from astropy.convolution import convolve_fft
import astropy.units as u
import numpy as np
from numpy import pi, exp
from scipy.special import gamma, gammaincinv, gammainc

# set up cosmology
cosmo = FlatLambdaCDM(H0=67.37, Om0=0.3147)


def fit_moffat(dist, amp, fwhm):
        beta = 3.
        gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
        norm = (beta-1)/(np.pi*gamma**2)
        return amp * norm * (1+(dist/gamma)**2)**(-1*beta)

max_radius = 10
diff = 0.02
INTSTEP = diff
tmp = np.arange(-max_radius, max_radius+diff, diff)
print(len(tmp))
INDEX = np.argmin(abs(tmp))
print("Middle row:", INDEX)
x = np.array([tmp for i in range(len(tmp))])
y = x.T
dist_map = np.sqrt(x**2+y**2)

small_fwhm_moffmap = fit_moffat(dist_map, amp=1, fwhm=1.2)
mid_fwhm_moffmap = fit_moffat(dist_map, amp=1, fwhm=1.3)
large_fwhm_moffmap = fit_moffat(dist_map, amp=1, fwhm=1.4)

small_fwhm_psf, large_fwhm_psf, mid_fwhm_psf = [], [], []
lutz_fwhm_psf = []
r_bins_psf_plot = np.arange(0, 10, 0.1)
for r in r_bins_psf_plot:
    xy = r/np.sqrt(2)
    here = np.sqrt((x-xy)**2+(y-xy)**2) < 0.75
    small_fwhm_psf.append(np.nansum(small_fwhm_moffmap[here])*diff**2)
    mid_fwhm_psf.append(np.nansum(mid_fwhm_moffmap[here])*diff**2)
    large_fwhm_psf.append(np.nansum(large_fwhm_moffmap[here])*diff**2)

small_fwhm_psf = np.array(small_fwhm_psf)
large_fwhm_psf = np.array(large_fwhm_psf)
mid_fwhm_psf = np.array(mid_fwhm_psf)

small_psf_integral = np.nansum(small_fwhm_psf[r_bins_psf_plot<=2.0])*0.1
mid_psf_integral = np.nansum(mid_fwhm_psf[r_bins_psf_plot<=2.0])*0.1
large_psf_integral = np.nansum(large_fwhm_psf[r_bins_psf_plot<=2.0])*0.1

psf_dict = {}
psf_dict["r/arcsec"] = r_bins_psf_plot
psf_dict["psf_mid"] = mid_fwhm_psf
psf_dict["psf_min"] = small_fwhm_psf
psf_dict["psf_max"] = large_fwhm_psf

ascii.write(psf_dict, "psf_range_profile.tab")
print("Wrote to psf_range_profile.tab")
