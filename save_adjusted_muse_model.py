from astropy.modeling.functional_models import Sersic2D
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9
from astropy.io import ascii
from astropy.convolution import convolve_fft
import astropy.units as u
import numpy as np
from numpy import pi, exp
from scipy.special import gamma, gammaincinv, gammainc

# set up cosmology
cosmo =  FlatLambdaCDM(H0=67.37, Om0=0.3147)


def fit_moffat(dist, amp, fwhm):
	beta = 3.
	gamma = fwhm/(2*np.sqrt(2**(1/beta) - 1))
	norm = (beta-1)/(np.pi*gamma**2)
	return amp * norm * (1+(dist/gamma)**2)**(-1*beta)

def b(n):
    # Normalisation constant
    return gammaincinv(2*n, 0.5)

def create_sersic_function(Ie, re, n):
    # Not required for integrals - provided for reference
    # This returns a "closure" function, which is fast to call repeatedly with different radii
    neg_bn = -b(n)
    reciprocal_n = 1.0/n
    f = neg_bn/re**reciprocal_n
    def sersic_wrapper(r):
        return Ie * exp(f * r ** reciprocal_n - neg_bn)
    return sersic_wrapper

def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = b(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*pi*n * exp(bn)/(bn**(2*n)) * g2n

def sersic_enc_lum(r, Ie, re, n):
    # luminosity enclosed within a radius r
    x = b(n) * (r/re)**(1.0/n)
    return sersic_lum(Ie, re, n) * gammainc(2*n, x)

def sersic(r, sb_e, r_e, n):
    b_n = 2*n-1/3.
    return sb_e * np.exp(-b_n*((r/r_e)**(1/n)-1))

lutz_bf_fh, lutz_bf_re, lutz_bf_n = 1448, 0.86, 2.8
lutz_best_fit = np.array([1448, 0.86, 2.8])
lutz_error = np.array([83, 0.11, 1.1])
lutz_best_fit_ps = 232
lutz_best_fit_ps_err = 50
lutz_best_fit_min = lutz_best_fit - lutz_error
lutz_best_fit_max = lutz_best_fit + lutz_error

kpc_per_arcsec_muse = cosmo.kpc_proper_per_arcmin(3.5)/60*u.arcmin/u.kpc
kpc_per_arcsec_muse

kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

max_radius = 10
diff = 0.02
INTSTEP = diff
tmp = np.arange(-max_radius, max_radius+diff, diff)
print(len(tmp))
INDEX = np.argmin(abs(tmp))
print("Middle row:", INDEX)
x = np.array([tmp for i in range(len(tmp))])
x_muse = x*kpc_per_arcsec_mid/kpc_per_arcsec_muse
y = x.T
y_muse = y*kpc_per_arcsec_mid/kpc_per_arcsec_muse
dist_map = np.sqrt(x**2+y**2)

def get_convolved_profile_sersic(fwhm, F_e, r_e, n, integrate_profile=False):
        kernel = fit_moffat(dist_map, amp=1, fwhm=fwhm)
        #kernel = gauss2d(dist_map, amp=1, fwhm=fwhm)

        sersic_integral = sersic_lum(1, re=r_e, n=n)
        lae_img = Sersic2D(amplitude=1, r_eff=r_e, n=n)(x=x_muse, y=y_muse) #sersic(dist_map, sb_e, r_e, n) #
        lae_img = lae_img*F_e/sersic_integral
        result = convolve_fft(lae_img, kernel)

        kernel_fiber = np.where(dist_map <= 0.75, 1, 0)
        kernel_fiber_int = np.nansum(kernel_fiber)*diff**2
        kernel_fiber = kernel_fiber / kernel_fiber_int
        result_2 = convolve_fft(result, kernel_fiber)
 

        result_dict = {"r": dist_map[INDEX, INDEX:],
                        "profile_raw": result[INDEX, INDEX:],
                        "profile_integrated": result_2[INDEX, INDEX:]}
        return result_dict

het_sersic = get_convolved_profile_sersic(1.3, *lutz_best_fit, integrate_profile=True)
het_sersic_min = get_convolved_profile_sersic(1.3, *lutz_best_fit_min, integrate_profile=True)
het_sersic_max = get_convolved_profile_sersic(1.3, *lutz_best_fit_max, integrate_profile=True)

het_lutz_model = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps, fwhm=1.3)+het_sersic["profile_integrated"]
het_lutz_model_min = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps-lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_min["profile_integrated"]
het_lutz_model_max = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps+lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_max["profile_integrated"]

het_lutz_model_integrated = np.nansum(het_lutz_model[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_min = np.nansum(het_lutz_model_min[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_max = np.nansum(het_lutz_model_max[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])

muse_dict = {}
muse_dict["r/arcsec"] = het_sersic["r"]
muse_dict["het_lutz_model"] = het_lutz_model
muse_dict["het_lutz_model_min"] = het_lutz_model_min
muse_dict["het_lutz_model_max"] = het_lutz_model_max

ascii.write(muse_dict, "muse_model_het_adjusted.tab")
print("Wrote to muse_model_het_adjusted.tab")

# now redshift 4-5
lutz_bf_fh, lutz_bf_re, lutz_bf_n = 931, 0.90, 3.3
lutz_best_fit = np.array([931, 0.9, 3.3])
lutz_error = np.array([82, 0.18, 1.9])
lutz_best_fit_ps = 150
lutz_best_fit_ps_err = 40
lutz_best_fit_min = lutz_best_fit - lutz_error
lutz_best_fit_max = lutz_best_fit + lutz_error

kpc_per_arcsec_muse = cosmo.kpc_proper_per_arcmin(4.5)/60*u.arcmin/u.kpc
kpc_per_arcsec_muse

kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

max_radius = 10
diff = 0.02
INTSTEP = diff
tmp = np.arange(-max_radius, max_radius+diff, diff)
print(len(tmp))
INDEX = np.argmin(abs(tmp))
print("Middle row:", INDEX)
x = np.array([tmp for i in range(len(tmp))])
x_muse = x*kpc_per_arcsec_mid/kpc_per_arcsec_muse
y = x.T
y_muse = y*kpc_per_arcsec_mid/kpc_per_arcsec_muse
dist_map = np.sqrt(x**2+y**2)

def get_convolved_profile_sersic(fwhm, F_e, r_e, n, integrate_profile=False):
        kernel = fit_moffat(dist_map, amp=1, fwhm=fwhm)
        #kernel = gauss2d(dist_map, amp=1, fwhm=fwhm)

        sersic_integral = sersic_lum(1, re=r_e, n=n)
        lae_img = Sersic2D(amplitude=1, r_eff=r_e, n=n)(x=x_muse, y=y_muse) #sersic(dist_map, sb_e, r_e, n) #
        lae_img = lae_img*F_e/sersic_integral
        result = convolve_fft(lae_img, kernel)

        kernel_fiber = np.where(dist_map <= 0.75, 1, 0)
        kernel_fiber_int = np.nansum(kernel_fiber)*diff**2
        kernel_fiber = kernel_fiber / kernel_fiber_int
        result_2 = convolve_fft(result, kernel_fiber)
 

        result_dict = {"r": dist_map[INDEX, INDEX:],
                        "profile_raw": result[INDEX, INDEX:],
                        "profile_integrated": result_2[INDEX, INDEX:]}
        return result_dict


het_sersic = get_convolved_profile_sersic(1.3, *lutz_best_fit, integrate_profile=True)
het_sersic_min = get_convolved_profile_sersic(1.3, *lutz_best_fit_min, integrate_profile=True)
het_sersic_max = get_convolved_profile_sersic(1.3, *lutz_best_fit_max, integrate_profile=True)

het_lutz_model = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps, fwhm=1.3)+het_sersic["profile_integrated"]
het_lutz_model_min = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps-lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_min["profile_integrated"]
het_lutz_model_max = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps+lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_max["profile_integrated"]

het_lutz_model_integrated = np.nansum(het_lutz_model[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_min = np.nansum(het_lutz_model_min[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_max = np.nansum(het_lutz_model_max[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])

muse_dict = {}
muse_dict["r/arcsec"] = het_sersic["r"]
muse_dict["het_lutz_model"] = het_lutz_model
muse_dict["het_lutz_model_min"] = het_lutz_model_min
muse_dict["het_lutz_model_max"] = het_lutz_model_max

ascii.write(muse_dict, "muse_model_het_adjusted_45.tab")
print("Wrote to muse_model_het_adjusted_45.tab")

# now redshift 5-6
lutz_bf_fh, lutz_bf_re, lutz_bf_n = 1002, 1.67, 6.5
lutz_best_fit = np.array([1002, 1.67, 6.5])
lutz_error = np.array([164, 0.86, 5.1])
lutz_best_fit_ps = 167
lutz_best_fit_ps_err = 34
lutz_best_fit_min = lutz_best_fit - lutz_error
lutz_best_fit_max = lutz_best_fit + lutz_error

kpc_per_arcsec_muse = cosmo.kpc_proper_per_arcmin(5.5)/60*u.arcmin/u.kpc
kpc_per_arcsec_muse

kpc_per_arcsec_mid = cosmo.kpc_proper_per_arcmin(2.5)/60*u.arcmin/u.kpc

max_radius = 10
diff = 0.02
INTSTEP = diff
tmp = np.arange(-max_radius, max_radius+diff, diff)
print(len(tmp))
INDEX = np.argmin(abs(tmp))
print("Middle row:", INDEX)
x = np.array([tmp for i in range(len(tmp))])
x_muse = x*kpc_per_arcsec_mid/kpc_per_arcsec_muse
y = x.T
y_muse = y*kpc_per_arcsec_mid/kpc_per_arcsec_muse
dist_map = np.sqrt(x**2+y**2)

def get_convolved_profile_sersic(fwhm, F_e, r_e, n, integrate_profile=False):
        kernel = fit_moffat(dist_map, amp=1, fwhm=fwhm)
        #kernel = gauss2d(dist_map, amp=1, fwhm=fwhm)

        sersic_integral = sersic_lum(1, re=r_e, n=n)
        lae_img = Sersic2D(amplitude=1, r_eff=r_e, n=n)(x=x_muse, y=y_muse) #sersic(dist_map, sb_e, r_e, n) #
        lae_img = lae_img*F_e/sersic_integral
        result = convolve_fft(lae_img, kernel)

        kernel_fiber = np.where(dist_map <= 0.75, 1, 0)
        kernel_fiber_int = np.nansum(kernel_fiber)*diff**2
        kernel_fiber = kernel_fiber / kernel_fiber_int
        result_2 = convolve_fft(result, kernel_fiber)
 

        result_dict = {"r": dist_map[INDEX, INDEX:],
                        "profile_raw": result[INDEX, INDEX:],
                        "profile_integrated": result_2[INDEX, INDEX:]}
        return result_dict


het_sersic = get_convolved_profile_sersic(1.3, *lutz_best_fit, integrate_profile=True)
het_sersic_min = get_convolved_profile_sersic(1.3, *lutz_best_fit_min, integrate_profile=True)
het_sersic_max = get_convolved_profile_sersic(1.3, *lutz_best_fit_max, integrate_profile=True)

het_lutz_model = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps, fwhm=1.3)+het_sersic["profile_integrated"]
het_lutz_model_min = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps-lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_min["profile_integrated"]
het_lutz_model_max = fit_moffat(het_sersic["r"], amp=lutz_best_fit_ps+lutz_best_fit_ps_err, fwhm=1.3)+het_sersic_max["profile_integrated"]

het_lutz_model_integrated = np.nansum(het_lutz_model[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_min = np.nansum(het_lutz_model_min[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])
het_lutz_model_integrated_max = np.nansum(het_lutz_model_max[het_sersic["r"]<=2.0])*(het_sersic["r"][1]-het_sersic["r"][0])

muse_dict = {}
muse_dict["r/arcsec"] = het_sersic["r"]
muse_dict["het_lutz_model"] = het_lutz_model
muse_dict["het_lutz_model_min"] = het_lutz_model_min
muse_dict["het_lutz_model_max"] = het_lutz_model_max

ascii.write(muse_dict, "muse_model_het_adjusted_56.tab")
print("Wrote to muse_model_het_adjusted_56.tab")
