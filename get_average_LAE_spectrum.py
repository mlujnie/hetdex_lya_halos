import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, join

from hetdex_tools.get_spec import get_spectra

import numpy as np
from astropy.io import ascii
from astropy.table import vstack
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import tables as tb

from hetdex_api.detections import Detections
from astropy.table import Table
line_catalog_class = Detections(curated_version='2.1.2')


from hetdex_api.config import HDRconfig
from hetdex_api.detections import Detections
from hetdex_api.elixer_widget_cls import ElixerWidget

line_catalog = line_catalog_class.return_astropy_table()

sources = ascii.read("../karls_suggestion/high_sn_sources.tab")
sources = sources[sources["mask"]==1]

print(len(sources), " sources in the tab.")

# get a single stacked spectrum from the LAEs

sources["redshift"] = sources["wave"]/1215.67-1

all_specs = []
broad_specs = []
narrow_specs = []
all_specs_loc = []
broad_specs_loc = []
narrow_specs_loc = []
N = len(sources)
for i in range(len(sources)):
    input_coords = SkyCoord(ra=[sources[i]["ra"]*u.deg], dec=[sources[i]["dec"]*u.deg])
    tmp = get_spectra(input_coords, ffsky=True)
    
    #tmp = line_catalog_class.get_spectrum(sources[i]["detectid"], )
    
    tmp["wave_rest"] = tmp["wavelength"]/(1+sources[i]["redshift"])

    ascii.write(tmp, "extracted_lae_spectra/ff_{}.dat".format(sources[i]["detectid"]))

    all_specs.append(tmp)
    if sources[i]["linewidth"] > 7:
        broad_specs.append(tmp)
    else:
        narrow_specs.append(tmp)

    tmp_loc = get_spectra(input_coords, ffsky=False)
    
    tmp_loc["wave_rest"] = tmp_loc["wavelength"]/(1+sources[i]["redshift"])

    ascii.write(tmp_loc, "extracted_lae_spectra/loc_{}.dat".format(sources[i]["detectid"]))
    Ns.append(len(wl_here[wl_here]))

    all_specs_loc.append(tmp_loc)
    if sources[i]["linewidth"] > 7:
        broad_specs_loc.append(tmp_loc)
    else:
        narrow_specs_loc.append(tmp_loc)

    if (i+1)%100==0:
        print(f"Done with {i+1}/{N}")

rest_specs = vstack(all_specs)
rest_specs_broad = vstack(broad_specs)
rest_specs_narrow = vstack(narrow_specs)

rest_specs_loc = vstack(all_specs_loc)
rest_specs_broad_loc = vstack(broad_specs_loc)
rest_specs_narrow_loc = vstack(narrow_specs_loc)

waves_rest = np.arange(np.nanmin(rest_specs["wave_rest"]), np.nanmax(rest_specs["wave_rest"])+2, 2)

stack = []
stack_broad = []
stack_narrow = []
stack_loc = []
stack_broad_loc = []
stack_narrow_loc = []
Ns = []
for w_min, w_max in zip(waves_rest[:-1], waves_rest[1:]):
    wl_here = (rest_specs["wave_rest"]>= w_min) & (rest_specs["wave_rest"]<w_max)
    wl_here_b = (rest_specs_broad["wave_rest"]>= w_min) & (rest_specs_broad["wave_rest"]<w_max)
    wl_here_n = (rest_specs_narrow["wave_rest"]>= w_min) & (rest_specs_narrow["wave_rest"]<w_max)
    stack.append(np.nanmedian(rest_specs["spec"][wl_here]))
    stack_broad.append(np.nanmedian(rest_specs_broad["spec"][wl_here_b]))
    stack_narrow.append(np.nanmedian(rest_specs_narrow["spec"][wl_here_n]))
    Ns.append(len(wl_here[wl_here]))

    wl_here = (rest_specs_loc["wave_rest"]>= w_min) & (rest_specs_loc["wave_rest"]<w_max)
    wl_here_b = (rest_specs_broad_loc["wave_rest"]>= w_min) & (rest_specs_broad_loc["wave_rest"]<w_max)
    wl_here_n = (rest_specs_narrow_loc["wave_rest"]>= w_min) & (rest_specs_narrow_loc["wave_rest"]<w_max)
    stack_loc.append(np.nanmedian(rest_specs_loc["spec"][wl_here]))
    stack_broad_loc.append(np.nanmedian(rest_specs_broad_loc["spec"][wl_here_b]))
    stack_narrow_loc.append(np.nanmedian(rest_specs_narrow_loc["spec"][wl_here_n]))

result_dict = {
	"wave_rest": waves_rest[:-1]+1,
	"stack_ff": stack,
	"stack_broad_ff": stack_broad,
	"stack_narrow_ff": stack_narrow,
	"stack_loc": stack_loc,
	"stack_broad_loc": stack_broad_loc,
	"stack_narrow_loc": stack_narrow_loc}

ascii.write(result_dict, "average_LAE_spectra_compare_skysubs.tab")
print("Done.")

