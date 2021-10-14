import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.stats import biweight_scale, biweight_location
import tables as tb

from scipy.interpolate import RegularGridInterpolator
INTERPOLATOR = RegularGridInterpolator

from scipy.ndimage import gaussian_filter
import glob
import pickle

# read in data
sources = ascii.read("../karls_suggestion/high_sn_sources_combined.tab")
sources = sources[sources["mask"]==1]

# get the luminosity
z_lae = (sources["wave"]/1215.67)-1
z_lae_err = (sources["wave_err"]/1215.67)
sources["redshift"] = z_lae
sources["luminosity"] = (sources["flux_213"])*1e-17*4*np.pi*(cosmo.luminosity_distance(sources["redshift"]).to(u.cm)/u.cm)**2
c = 3*10**5 # km/s
doppler_v_of = c * sources["linewidth"] / sources["wave"]
sources["linewidth_km/s"] = doppler_v_of

total_mask = np.ones(len(sources), dtype=bool)

SN_65 = True #(args.sn_65 == 'True')
if SN_65:
	print('Including only S/N>6.5.')
	total_mask = total_mask * (sources['sn']>6.5)
else:
	print('Including all S/N>5.5.')

narrow_lines = sources["linewidth_km/s"] < 1000/2.35
low_luminosity = sources["luminosity"] < 10**43

broad_line_sample = ~narrow_lines
print('1. The broad line sample contains {} sources.'.format(len(broad_line_sample[broad_line_sample])))

narrow_line_high_L_sample = narrow_lines * (~low_luminosity)
print('2. The narrow line, high L sample contains {} sources.'.format(len(narrow_line_high_L_sample[narrow_line_high_L_sample])))

narrow_line_low_L_sample = narrow_lines * low_luminosity
print('3. The narrow line, low L sample contains {} sources.'.format(len(narrow_line_low_L_sample[narrow_line_low_L_sample])))

sample = 3
SAMPLE_1 = sample == 1
SAMPLE_2 = sample == 2
SAMPLE_3 = sample == 3
if (not SAMPLE_1) * (not SAMPLE_2) * (not SAMPLE_3):
	logging.error('The --sample option must have a 1, 2, or 3. Cancelling this script.')
	sys.exit()

if SAMPLE_1:
	print('Using broad line sample.')
	sources = sources[total_mask * broad_line_sample]
elif SAMPLE_2:
	print('Using narrow line, high L sample.')
	sources = sources[total_mask * narrow_line_high_L_sample]
elif SAMPLE_3:
	print('Using narrow line, low L sample.')
	sources = sources[total_mask * narrow_line_low_L_sample]

N = len(sources)
print('len(sources):', N)

# new bogus LAEs
bogus_list = []
i=0
for source in sources:
	detectid = source["detectid"]
	lae_file_pattern = "/scratch/05865/maja_n/radial_profiles/bogus_laes_newflag_samelambda/bogus_?_{}.dat".format(detectid)
	lae_files = glob.glob(lae_file_pattern)
	for lae_file in lae_files:
		try:
			lae_tab = ascii.read(lae_file)
			lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
			lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
			bogus_list.append(lae_tab)
			i+=1
			if i%100 == 0:
				print(f"Finished {i}/{N*3}")
		except Exception as e:
			print("Failed to read "+lae_file)
	   
bogus_tab = vstack(bogus_list)
bogus_tab["mask_7"] = bogus_tab["mask_7"].astype(bool)
bogus_tab["mask_10"] = bogus_tab["mask_10"].astype(bool)

bogus_tab = bogus_tab[bogus_tab["flux"]!=0]
fiber_area = np.pi*0.75**2
fiber_area

std = np.nanstd(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area)
mid = np.nanmedian(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area)#, std, bstd
bstd = biweight_scale(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area)
perc_16 = np.nanpercentile(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area, 16)
perc_84 = np.nanpercentile(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area, 100-16)

fiber_area = np.pi*0.75**2
bins= np.arange(-2.5, 2.5, 0.02)

hist1, bins1 = np.histogram(bogus_tab["flux_troughsub"]/fiber_area, bins=bins)
hist2, bins2 = np.histogram(bogus_tab["flux_troughsub"][bogus_tab["mask_7"]]/fiber_area, bins=bins)

histograms = {
	"bin_edges": bins2,
	"all": hist1,
	"masked": hist2,
	"median": mid,
	"std": std,
	"biweight_scale": bstd,
	"perc_16": perc_16,
	"perc_84": perc_84
}

with open("bogus_histogram_samelambda.pickle", "wb") as fi:
	pickle.dump(histograms, fi)

print("Wrote to bogus_histogram_samelambda.pickle")
