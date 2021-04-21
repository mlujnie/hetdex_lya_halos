from astropy.io import ascii
from astropy.table import vstack
import numpy as np
import glob
import pickle

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
print("Number of sources: ", len(sources))

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
		lae_tab["ra_diff"] = (lae_tab["ra"]-source["ra"]) * np.cos(source["dec"]*np.pi/180)
		lae_tab["dec_diff"] = lae_tab["dec"]-source["dec"]
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

ras = np.arange(-50, 52, 1.)
ra_dec_grid = np.array([[(ra, dec) for ra in ras] for dec in ras])

long_tab = long_tab[(long_tab["flux_contsub"]!=0)&(long_tab["mask_7"])] # take care to always include the mask for the LAEs!

# divide by fiber_area to get a surface brightness!
fiber_area = np.pi*0.75**2
long_tab["flux_contsub"] /= fiber_area

flux_grid = np.zeros((ras.size-1, ras.size-1))
mean_grid = np.zeros((ras.size-1, ras.size-1))
for i in range(len(flux_grid)-1):
	for j in range(len(flux_grid)-1):
		ra_min, ra_max = ras[i]/3600., ras[i+1]/3600.
		dec_min, dec_max = ras[j]/3600., ras[j+1]/3600.
		here = (long_tab["ra_diff"]>=ra_min)&(long_tab["ra_diff"]<ra_max)&(long_tab["dec_diff"]>=dec_min)&(long_tab["dec_diff"]<dec_max)
		flux_grid[i,j] = np.nanmedian(long_tab["flux_contsub"][(long_tab["ra_diff"]>=ra_min)&(long_tab["ra_diff"]<ra_max)&(long_tab["dec_diff"]>=dec_min)&(long_tab["dec_diff"]<dec_max)])
		mean_grid[i,j] = np.nanmean(long_tab["flux_contsub"][(long_tab["ra_diff"]>=ra_min)&(long_tab["ra_diff"]<ra_max)&(long_tab["dec_diff"]>=dec_min)&(long_tab["dec_diff"]<dec_max)])

pythonObject = (ras, flux_grid, mean_grid)
with open("lae_2d_grid.pkl", "wb") as pickleDestination:
	pickle.dump(pythonObject, pickleDestination)
