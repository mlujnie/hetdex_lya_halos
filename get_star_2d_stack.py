from astropy.io import ascii
from astropy.table import vstack
import numpy as np
import glob
import pickle

star_list = []
star_sources = ascii.read("star_gaia_tab.tab")
i=0
N = 3795

for lae_file in glob.glob("../radial_profiles/stars_skymask/star_*.dat"):
	try:
		detectid = lae_file.split("/")[-1][5:-4]
		this_star = star_sources[star_sources["detectid"]==int(detectid)]
		this_ra, this_dec = this_star["ra"], this_star["dec"]
		
		lae_tab = ascii.read(lae_file)
		lae_tab["mask_7"] = np.array(lae_tab["mask_7"]=="True").astype(int)
		lae_tab["mask_10"] = np.array(lae_tab["mask_10"]=="True").astype(int)
		lae_tab["detectid"] = [detectid for j in range(len(lae_tab))]
		lae_tab["ra_diff"] = (lae_tab["ra"] - this_ra) * np.cos(this_dec * np.pi/180.)
		lae_tab["dec_diff"] = lae_tab["dec"] - this_dec
		star_list.append(lae_tab)
		i+=1
		if i%100 == 0:
			print(f"Finished {i}/{N}")
	except Exception as e:
		print(e)
		pass


star_tab= vstack(star_list)
star_tab["mask_7"] = star_tab["mask_7"].astype(bool)
star_tab["mask_10"] = star_tab["mask_10"].astype(bool)

ras = np.arange(-50, 52, 1.)
ra_dec_grid = np.array([[(ra, dec) for ra in ras] for dec in ras])

long_tab = star_tab[star_tab["flux"]!=0] #long_tab[long_tab["flux"]!=0]

flux_grid = np.zeros((ras.size-1, ras.size-1))
for i in range(len(flux_grid)-1):
	for j in range(len(flux_grid)-1):
		ra_min, ra_max = ras[i]/3600., ras[i+1]/3600.
		dec_min, dec_max = ras[j]/3600., ras[j+1]/3600.
		here = (long_tab["ra_diff"]>=ra_min)&(long_tab["ra_diff"]<ra_max)&(long_tab["dec_diff"]>=dec_min)&(long_tab["dec_diff"]<dec_max)
		flux_grid[i,j] = np.nanmedian(long_tab["flux"][(long_tab["ra_diff"]>=ra_min)&(long_tab["ra_diff"]<ra_max)&(long_tab["dec_diff"]>=dec_min)&(long_tab["dec_diff"]<dec_max)])#long_tab["mask_7"]])

pythonObject = (ras, flux_grid)
with open("star_2d_grid.pkl", "wb") as pickleDestination:
	pickle.dump(pythonObject, pickleDestination)
