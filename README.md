# hetdex_lya_halos

1. step: save surface brightness around LAEs and bogus LAEs (random areas), and average flux of stars with save_laes.py, save_bogus.py and save_stars.py

2. step: save radial profiles: save_radial_profiles.py

3. step: save background values: save_random_sky_values.py

4. step: save wavelength-shifted radial profiles: save_lutz_errors.py, then get the standard deviation of those: get_lutz_error.py

5. save stacked radial profile of stars: save_star_stack.py

6. prepare more things for the plots: save_adjusted_muse_model.py save_adjusted_psf.py get_average_LAE_halo_spectrum.py get_bogus_histogram.py get_lae_2d_stack.py playground_residuals.py
