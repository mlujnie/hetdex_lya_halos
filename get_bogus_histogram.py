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

sources = ascii.read("high_sn_sources.tab")
sources = sources[sources["mask"]==1]
print(len(sources))

# new bogus LAEs
bogus_list = []
i=0
N = len(sources)*2
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
    if True:
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
            print(e)
        
bogus_tab = vstack(bogus_list)
bogus_tab["mask_7"] = bogus_tab["mask_7"].astype(bool)
bogus_tab["mask_10"] = bogus_tab["mask_10"].astype(bool)

bogus_tab = bogus_tab[bogus_tab["flux"]!=0]
fiber_area = np.pi*0.75**2
fiber_area

std = np.nanstd(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area)
mid = np.nanmedian(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area)#, std, bstd
bstd = biweight_scale(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area)
perc_16 = np.nanpercentile(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area, 16)
perc_84 = np.nanpercentile(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area, 100-16)

fiber_area = np.pi*0.75**2
bins= np.arange(-2.5, 2.5, 0.02)

hist1, bins1 = np.histogram(bogus_tab["flux_contsub"]/fiber_area, bins=bins)
hist2, bins2 = np.histogram(bogus_tab["flux_contsub"][bogus_tab["mask_7"]]/fiber_area, bins=bins)

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

with open("bogus_histogram.pickle", "wb") as fi:
	pickle.dump(histograms, fi)

print("Wrote to bogus_histogram.pickle")
