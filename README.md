# lhires_pipeline
A pipeline for processing LHIRES-III spectroscopic data

The pipeline runs as a Jupyter notebook. Future implementation will allow it to run as a Python script on the terminal.


data_dir (str): full path of the directory containing the raw frames
output_dir  (str): full path of the directory to store the processed frames and products (plots) 

file_extension (str, default: '.fit'): extension of the raw frames (for LHIRES-III, the default is '.fit')

bias_prefix (str, default: 'bias_'):          prefix of the bias frames (the default naming convention is <prefix + 0001 + file_extension>)
flat_prefix (str, default: 'dflat_300s_'):    prefix of the flat frames
arc_prefix  (str, default: 'hear_300s_'):     prefix of the arc lamp frames
sci_prefix  (str, default: 'etaCar_300s_'):   prefix of the science frames 
std_prefix  (str, default: 'HIP54830_600s_'): prefix of the standard star frames 

fileout_bias (str, default: 'master_bias'): name for the master bias frame
fileout_flat (str, default: 'master_flat'): name for the master flat frame (used for illumination correction, not implemented)
fileout_norm_flat (str, default: 'master_norm_flat'): name for the master normalized flat frame (used for pixel-to-pixel correction)

sci_name (str, default: 'eta Carinae'): Name of the science target. The code will query SIMBAD to search for RA and Dec information
std_name (str, default: 'HIP54830'):    Name of the standard star.  The code will query SIMBAD to search for RA and Dec information

The HeAr peaks were selected for the wavelength interval (4545-5035 A), if using other setup, please modify the values below. 
HeAr_peak_wavelengths (numpy.array, float, values in Angstrom): wavelengths corresponding to the peak of HeAr lines used for wavelength calibration
HeAr_peak_pixels      (numpy.array, integer, values in pixel):  approximate pixel position of the peaks defined in 'HeAr_peak_wavelengths'

n_pix_step (integer, default=10): model spatial distortion along the slit for every 'n_pix_step' pixels
ndeg_distortion (integer, default=2): use a 'ndef_distortion'-th polynomial model to fit the spatial distortion
ndeg_wavesolution (integer, default=3): use a 'ndeg_wavesolution'-th polynomial model to fit the wavelength solution

save_plots (boolean, default=True): save plots as files.
figsize (list, default=(15,5)): default size for the plots.
fontsize (integer, default=12): font size for the plots
plot_extension (str, default='.png'): extension for the plots.
