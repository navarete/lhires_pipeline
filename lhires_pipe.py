import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.stats import sigma_clip, mad_std
from astropy.time import Time
from astroquery.simbad import Simbad
from ccdproc import Combiner, ImageFileCollection, subtract_bias, flat_correct, combine
from scipy.constants import c as c_si  # speed of light in m/s
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit


# --- Utility Function ---
def load_fits_data(directory, prefix, extension='.fit', return_file_names=False):
    """
    Load FITS files from a directory whose filenames start with a given prefix.
    
    Parameters:
        directory (str): Path to the directory containing FITS files.
        prefix (str): Prefix to match filenames (e.g., 'bias', 'flat', 'science').

    Returns:
        list of CCDData: List of loaded CCDData objects.
    """
    files = sorted([f for f in os.listdir(directory) if f.endswith(extension) and f.startswith(prefix)])
    filepaths = [os.path.join(directory, f) for f in files]
    if return_file_names:
        return [CCDData.read(f, unit='adu') for f in filepaths], files
    else:
        return [CCDData.read(f, unit='adu') for f in filepaths]

def plot_data(image, percent_display=None, title='', xlabel='x axis (pixel)', ylabel='y axis (pixel)', colorscale_label='adu', 
              pad_colorbar=0.01, figsize=(20,4), fontsize=12, output=None):

    plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = fontsize
    if percent_display is None:
        img = plt.imshow(image, aspect='auto', origin='bottom')
    else:
        displ_range = np.percentile(image, percent_display)
        img = plt.imshow(image, aspect='auto', origin='bottom',vmin=displ_range[0],vmax=displ_range[1])
        percent_display
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(img, pad=pad_colorbar, label=colorscale_label)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=300)
        
    plt.show()

# flat field functions
    
def inv_median(a):
    """ Define weight for flat field combining """
    return 1 / np.median(a)

def process_frames(ccddata_frames, bias=None, flat=None, sigma_clip=5, scale=None):
    """ performs bias subtraction, flat-field correction and combine frames using avsigclip method."""
    # start list to store the reduced images
    reduced_frames = []
    
    # loop into the raw images
    for i, frame in enumerate(ccddata_frames):
        # bias subtract
        if bias is not None:
            reduced_frame = subtract_bias(frame, bias)
            reduced_frame.header['BIAS_SUB'] = True
            reduced_frame.header.comments['BIAS_SUB'] = 'Bias subtraction was performed (T or F)'
        else:
            reduced_frame = frame
            reduced_frame.header['BIAS_SUB'] = False
            reduced_frame.header.comments['BIAS_SUB'] = 'Bias subtraction was performed (T or F)'
        if flat is not None:
            # flat correction - using the normalized flat, not the illumination correction!
            reduced_frame = flat_correct(reduced_frame, flat)
            reduced_frame.header['FLAT_COR'] = True
            reduced_frame.header.comments['FLAT_COR'] = 'Flat-field correction was performed (T or F)'
        else:
            reduced_frame.header['FLAT_COR'] = False
            reduced_frame.header.comments['FLAT_COR'] = 'Flat-field correction was performed (T or F)'
            
        # store the processed frames into a list
        reduced_frames.append(reduced_frame)
    
    # combine with average and sigma clipping method
    master_frame = combine(reduced_frames, method='average', sigma_clip=True, scale=scale,
                           sigma_clip_low_thresh=sigma_clip, sigma_clip_high_thresh=sigma_clip,
                           sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)
    
    # add header keywords
    master_frame.header['COMBINE'] = 'avsigclip'
    master_frame.header.comments['COMBINE'] = 'Image combine method'
    master_frame.header['NCOMBINE'] = len(reduced_frames)
    master_frame.header.comments['NCOMBINE'] = 'Number of combined images'
    
    return master_frame
    

# wavelength calibration functions

def gaussian_kernel(x, amp, mu, sigma, offset):
    """ Gaussian kernel to search for arc lines """
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

def refine_arc_peak_positions(spectrum, initial_guesses, window=10, plot_fits=False, n_col=None, output=None):
    """
    Refine the positions of arc lines from initial pixel guesses by fitting Gaussians.
    
    Parameters:
        spectrum (np.ndarray): 1D array of arc lamp flux.
        initial_guesses (list): Approximate pixel locations of arc peaks.
        window (int): Half-width of region to fit around each guess.
        plot_fits (bool): If True, plots the fitted region.

    Returns:
        refined_positions (np.ndarray): Subpixel centroid positions of each arc line.
    """
    refined_positions = []
    n_peaks = len(initial_guesses)
    if n_col is None:
        n_col = n_peaks if n_peaks <= 4 else 4
    n_row = int(n_peaks / n_col)
    if n_col * n_row < n_peaks:
        n_row += 1
    
    if plot_fits:
        plt.figure(figsize=(n_col * 4, n_row * 3))
    
    for n, guess in enumerate(initial_guesses):
        # Define local fitting window
        x_fit = np.arange(guess - window, guess + window + 1)
        y_fit = spectrum[x_fit]

        # Initial parameters: amplitude, center, width, offset
        amp0 = y_fit.max() - y_fit.min()
        mu0 = guess
        sigma0 = 2.0
        offset0 = y_fit.min()

        try:
            popt, _ = curve_fit(gaussian_kernel, x_fit, y_fit, p0=[amp0, mu0, sigma0, offset0])
            refined_mu = popt[1]
            fwhm = 2.355 * popt[2]
            refined_positions.append(refined_mu)

            if plot_fits:
                plt.subplot(n_row,n_col,n+1)
                plt.plot(x_fit, y_fit, 'k.')
                plt.plot(x_fit, gaussian_kernel(x_fit, *popt), 'r-', label='Fit')
                plt.axvline(refined_mu, color='blue', linestyle='--', label=r'x$_{peak}$, $\Delta$x='+f'{refined_mu:.2f}, {fwhm:.2f}')
                plt.title(f'Line near x={guess}')
                plt.grid(True)
                if n == 0:
                    plt.ylabel('Flux')
                plt.xlabel('Pixel')
                plt.legend()
                
        except RuntimeError:
            print(f"Fit failed near x={guess}")
            refined_positions.append(guess)  # fallback to original guess

    if plot_fits:
        plt.suptitle("Refine peak positions for the Arc lamp spectrum",y=1.0)
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()            
            
    return np.array(refined_positions)

def extract_1d_arc_spectrum(arc_frame, row_range=(100, 110)):
    """
    Extract 1D arc spectrum by summing rows across a spatial region.
    
    Parameters:
        arc_frame (CCDData): The 2D arc frame.
        row_range (tuple): Tuple (y1, y2) defining rows to extract.

    Returns:
        1D numpy array: Extracted arc spectrum (flux vs pixel).
    """
    data = arc_frame.data
    spectrum_1d = np.sum(data[row_range[0]:row_range[1], :], axis=0)
    return spectrum_1d

# fit spatial distortion along the slit
def fit_spatial_distortion(image, peak_positions_in_pixel, n_pix_step=10, ndeg=2, n_col=None, output=None):
    
    # use the 'refined_peak_positions_in_pixel' value to trace the lines along the slit
    n_spat, n_spec = image.shape
    spat_rows = np.arange(n_pix_step, n_spat, n_pix_step)

    # get number of spatial rows
    n_spat_rows = len(spat_rows)

    # now get the refined peak positions for every line and across the rows in 'spat_rows'
    peak_refined_rows = []
    for row in spat_rows:
        # set center and window on y-axis to extract the spectrum
        y_center = row
        row_range = (y_center - 1, y_center + 1 )
        arc_1d = extract_1d_arc_spectrum(image, row_range=row_range)
        # refine x_peak positions for each row
        peak_refined_row     = refine_arc_peak_positions(arc_1d, peak_positions_in_pixel, window=30, plot_fits=False)
        # store the data into a list
        peak_refined_rows.append(peak_refined_row)
    # convert list to numpy array
    peak_refined_rows=np.array(peak_refined_rows, dtype='float32')

    # get number of spectral lines to fit
    n_lines = len(peak_positions_in_pixel)
    
    n_plots = n_lines
    if n_col is None:
        n_col = n_plots if n_plots <= 4 else 4
    n_row = int(n_plots / n_col)
    if n_col * n_row < n_plots:
        n_row += 1 
    
    # now plot
    plt.figure(figsize=(3*n_col,3*n_row))
    

    line_fit = []
    for n in np.arange(n_lines):
        # set (x,y) for each line - we want f(y)=x!
        y = peak_refined_rows[:,n]
        x = spat_rows
        # start the fitter
        linfitter = LinearLSQFitter()
        spatial_model = Polynomial1D(degree=ndeg)
        fit_spat = linfitter(model=spatial_model, x=x, y=y)
        # store the polynomial fits
        line_fit.append(fit_spat.parameters)
        # get yfit
        yfit = fit_spat(x) #* u.nm
        #plot results
        plt.subplot(n_row,n_col,n+1)
        plt.plot(yfit,x,color='red',label='Model')
        plt.plot(y,x,'+',label='Data')
        plt.xlabel('Dispersion (pixel)')
        plt.title(f'Line peak at x={peak_positions_in_pixel[n]:.1f}')
        if n == 0:
            plt.ylabel('Along slit (pixel)')
    plt.legend()
    plt.suptitle('Spatial distortion fitting for selected HeAr lines along the slit', y=1.01)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=300)
    plt.show()

    # now, store the polynomial fits for each line
    polyfits_x_along_y = []
    for row in spat_rows:
        # set (x,y) for each line
        y = row
        # for each line, construct the polynomial mode to evaluate x_peak as a function of the y-axis 
        for coeffs in line_fit:
            # Reconstruct the model
            x_peak_model = models.Polynomial1D(degree=len(coeffs)-1)
            x_peak_model.parameters = coeffs
            polyfits_x_along_y.append(x_peak_model)

    polyfits_x_along_y = np.array(polyfits_x_along_y)
    polyfits_x_along_y = np.reshape(polyfits_x_along_y, (n_spat_rows, -1))

    return spat_rows, polyfits_x_along_y

def correct_spatial_distortion(ccddata, y_positions, x_y_fits, reference_peak_positions, y_ref=None, show_plot=True, figsize=(12,12), output=None):
    """
    Rectify 2D image using a set of x(y) distortion curves at given x positions.
    
    Parameters:
        ccddata : 2D numpy array (CCDData type)
            The distorted input image [ny, nx].
        y_positions : array-like
            The x positions at which the x(y) functions were measured.
        x_y_fits : list of Polynomial1D models
            The distortion curves (x as a function of y) for each x_position.
        y_ref : float
            Reference y value to align all x(y) to (default: center of image).
    
    Returns:
        rectified_ccddata : 2D numpy array (CCDData)
            The distortion-corrected image.
    """
    ny, nx = ccddata.shape
    if y_ref is None:
        y_ref = ny // 2
    
    y_vals = np.arange(ny)
    x_vals = np.arange(nx)

    # Step 1: Evaluate x(y) for each given x_position
    distortion_curves = np.zeros((len(y_positions), ny))  # [n_sample_y, ny]
    #print(distortion_curves.shape)
    
    for i, fit in enumerate(x_y_fits):
        distortion_curves[i, :] = fit(y_vals)

    # Step 2: Interpolate across all x positions to make a full 2D distortion map
    x_shift_map = np.zeros((ny, nx))
    for y in range(ny):
        interp = interp1d(y_positions, distortion_curves[:, y], kind='cubic', fill_value='extrapolate')
        distorted_x = interp(x_vals)
        # Reference: what x value should this trace have at y_ref?
        ref_interp = interp1d(y_positions, [fit(y_ref) for fit in x_y_fits], kind='cubic', fill_value='extrapolate')
        ref_x = ref_interp(x_vals)
        x_shift_map[y, :] = ref_x - distorted_x

    # Step 3: Apply the shift to resample the image
    y_coords, x_coords = np.indices(ccddata.shape)
    corrected_x = x_coords - x_shift_map
    
    rectified_ccddata = map_coordinates(ccddata, [y_coords, corrected_x], order=1, mode='nearest')

    
    if show_plot:
        xmin=0
        xmax=ccddata.shape[1]

        plt.figure(figsize=figsize)
        
        plt.subplot(4,1,1)
        plt.imshow(ccddata, aspect='auto', origin='bottom', vmin=0, vmax=1000)
        for x in reference_peak_positions:
            plt.axvline(x, ls='--', color='black')
        plt.xlim(xmin,xmax)
        plt.title("Original HeAr lamp")
        
        plt.subplot(4,1,2)
        plt.imshow(rectified_ccddata, aspect='auto', origin='bottom', vmin=0, vmax=1000)
        for x in reference_peak_positions:
            plt.axvline(x, ls='--', color='black')
        plt.xlim(xmin,xmax)
        plt.title("Distortion corrected HeAr lamp")

        plt.subplot(4,1,3)
        y = ccddata.data
        yps = [25,150,275]
        for ypp in yps:
            yp = y[ypp,:]
            plt.plot(np.arange(len(yp)),yp, label=f"y={ypp} (original)")
        for x in reference_peak_positions:
            plt.axvline(x, ls='--', color='black')
        plt.xlim(900,1500)
        plt.ylim(0,1500)
        plt.legend()
        plt.title("Original HeAr lamp")

        plt.subplot(4,1,4)
        y = rectified_ccddata
        for ypp in yps:
            yp = y[ypp,:]
            plt.plot(np.arange(len(yp)),yp, label=f"y={ypp} (corrected)")
        for x in reference_peak_positions:
            plt.axvline(x, ls='--', color='black')
        plt.xlim(900,1500)
        plt.ylim(0,1500)
        plt.legend()
        plt.title("Distortion corrected HeAr lamp")
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    # save header
    header_out = ccddata.header
        
    return CCDData(rectified_ccddata, unit='adu', header=header_out)

# wavelength solution
def wavelength_calibration(x, y, ndeg=3, show_plot=True, figsize=(12,5), output=None):

    # start the fitter
    linfitter = LinearLSQFitter()
    spatial_model = Polynomial1D(degree=ndeg)
    fit_wavelength = linfitter(model=spatial_model, x=x, y=y)
    wave_x = fit_wavelength(x)
    # compute RMS of the fitting
    residuals = y-wave_x
    rms = np.sqrt(np.sum(abs(residuals)**2)/len(residuals))
    
    if show_plot:
        plt.figure(figsize=figsize)
        
        plt.subplot(2,1,1)
        plt.plot(x,y,'+',label='Measurements')
        plt.plot(x,wave_x,color='red', alpha=0.5, label=f'Model (deg={ndeg})')
        plt.ylabel(r'Wavelength ($\AA$)')
        plt.xlabel('Dispersion axis (pixel)')
        plt.legend()
        # plot residuals
        plt.subplot(2,1,2)
        plt.plot(y,residuals,'+',label=fr'RMS={rms:.6f} $\AA$' )
        plt.axhline(0,ls='--',color='grey')
        plt.legend()
        plt.ylabel(r'Residuals ($\AA$)')
        plt.xlabel(r'Wavelength ($\AA$)')
        # make y-axis centered at 0.
        ymax = np.max(abs(residuals))*1.1
        plt.ylim(-ymax,ymax)
        plt.suptitle("Wavelength calibration", va='bottom')
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()
        
    return fit_wavelength, rms

# 1d spectrum extraction
def extract_spectrum_1d(ccddata, extract_percentile=None, show_plot=True, figsize=(12,5), output=None):
    """
    Extract 1D spectrum by summing spatial rows.
    
    Parameters:
        ccddata (CCDData): The 2D science frame.
        extract_percentile (float): Set the flux limit (from 0 to 1) to extract the spectrum based on the stellar profile.
                                    If None, sum all rows.

    Returns:
        1D numpy array: The extracted stellar spectrum.
    """
    data   = ccddata.data
    header = ccddata.header
    
    x_axis = np.arange(data.shape[1])
    y_axis = np.arange(data.shape[0])
    
    if extract_percentile is None:
        spectrum_1d = np.sum(data, axis=0)
        xi, xf = 1, np.max(y_axis)
        x_peak_profile = int(0.5*data.shape[0])
        
    else:
        stellar_profile = data.mean(axis=1)
        stellar_profile -= np.min(stellar_profile)
        stellar_profile /= np.nanmax(stellar_profile)

        # get peak position
        x_peak_profile = y_axis[np.argmax(stellar_profile)]
        x_left  = y_axis < x_peak_profile
        x_right = y_axis > x_peak_profile
        # now find where the flux reaches the 'extract_percentile' limits in both sides of the peak profile
        xi = y_axis[x_left].flat[np.abs(stellar_profile[x_left] - extract_percentile).argmin()]
        xf = y_axis[x_right].flat[np.abs(stellar_profile[x_right] - extract_percentile).argmin()]

    spectrum_1d = np.sum(data[xi:xf,:], axis=0)

    if show_plot and extract_percentile is not None:
        plt.figure(figsize=figsize)
        plt.subplot(2,1,1)
        plt.plot(y_axis,stellar_profile)
        plt.axvline(xi, ls='--', color='black')
        plt.axvline(x_peak_profile, ls='--', color='red', label=r'x$_{peak}$='+f'{x_peak_profile}')
        plt.axvline(xf, ls='--', color='black', label=f'Extract_Percentile={extract_percentile}')
        plt.xlabel('Slit direction (pixel)')
        plt.ylabel('Normalized stellar profile')
        plt.xlim(np.nanmin(y_axis),np.nanmax(y_axis))
        plt.legend()

        #wspectrum_1d = np.sum(data * stellar_profile[:, np.newaxis], axis=0)
        plt.subplot(2,1,2)
        plt.plot(x_axis,spectrum_1d)
        plt.xlabel('Dispersion direction (pixel)')
        plt.ylabel('Extracted flux')
        plt.xlim(np.nanmin(x_axis),np.nanmax(x_axis))
        #plt.plot(x_axis,wspectrum_1d)
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()
        
    if show_plot and extract_percentile is  None:
        plt.figure(figsize=figsize)
        plt.plot(x_axis,spectrum_1d)
        plt.xlabel('Dispersion direction (pixel)')
        plt.ylabel('Extracted flux')
        plt.xlim(np.nanmin(x_axis),np.nanmax(x_axis))
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()
            
    # add header keywords
    header['EXTR_CEN'] = x_peak_profile
    header.comments['EXTR_CEN'] = 'Central position along the slit for extracting the flux'
    header['EXTR_WID'] = abs(xf-xi)
    header.comments['EXTR_WID'] = 'Width of extraction'
    
    return CCDData(spectrum_1d, unit='adu', header=header)

def interpolate_wavelength_axis(flux, wavelength, dispersion=None, show_plot=False, figsize=(12,4)):
    """
    Create a linear wavelength grid from a non-linear wavelength array.
    """
    wave_min = wavelength.min()
    wave_max = wavelength.max()

    # Estimate the mean delta lambda from original
    deltas = np.diff(wavelength)
    if dispersion is None:
        delta_lambda = np.nanmin(deltas)
    else:
        delta_lambda = dispersion
        
    wave_linear = np.arange(wave_min, wave_max, delta_lambda)
    
    f_interp = interp1d(wavelength, flux.data, kind='cubic', fill_value='extrapolate')
    flux_interp = f_interp(wave_linear)
    
    header = flux.header
    
    # Wavelength solution keywords
    crpix1 = 1  # Reference pixel (1-based in FITS)
    crval1 = wave_linear[0]  # Wavelength at reference pixel
    cdelt1 = wave_linear[1] - wave_linear[0]  # Dispersion (assumes linear)
    
    header['CRPIX1'] = crpix1
    header['CRVAL1'] = crval1
    header['CDELT1'] = cdelt1
    header['CTYPE1'] = 'Wavelength'
    header['CUNIT1'] = 'Angstrom'

    if show_plot:
        plt.figure(figsize=figsize)
        plt.plot(wavelength,  flux.data,   alpha=0.75, label='original')
        plt.plot(wave_linear, flux_interp, alpha=0.75, label='interpolated')
        plt.xlim(np.nanmin(wavelength),np.nanmax(wavelength))
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel(r'Flux (ADU)')
        plt.legend()
        plt.show()
    
    return CCDData(flux_interp, unit='adu', header=header), wave_linear


# heliocentric correction

def get_coordinates_from_simbad(target, show_messages=False):
    # Query SIMBAD for the object
    result = Simbad.query_object(target)
    
    if result is None:
        print(f"Target '{target}' not found in SIMBAD.")
        return None

    # Extract RA and DEC in degrees
    ra = result['RA'][0]       # in sexagesimal (HH MM SS)
    dec = result['DEC'][0]     # in sexagesimal (DD MM SS)

    # Convert to decimal degrees using astropy
    coord = SkyCoord(f"{ra} {dec}", unit=("hourangle", "deg"))
    
    # Print coordinates
    if show_messages:
        print(f"Target: {target}")
        print(f"RA  (sexagesimal): {ra}")
        print(f"Dec (sexagesimal): {dec}")
    
    
        print(f"RA  (deg): {coord.ra.deg:.6f}")
        print(f"Dec (deg): {coord.dec.deg:.6f}")
    
    return coord.ra.deg, coord.dec.deg

def apply_heliocentric_correction(wavelength, vhelio):
    """
    Shift the wavelength grid to heliocentric rest frame.

    Parameters:
        wavelength : array-like
            Original wavelength grid in Angstrom.
        vhelio : float
            Heliocentric velocity correction in km/s.

    Returns:
        wavelength_corrected : array-like
            Corrected wavelength grid.
    """
    c_km_s = c_si / 1000  # convert to km/s
    shift_factor = 1 - (vhelio / c_km_s)
    wavelength_corrected = wavelength * shift_factor
    return wavelength_corrected

def get_wavelenght_axis(ccddata):
    
    header = ccddata.header # FITS header

    # Extract WCS-related keywords
    crval1 = header.get('CRVAL1')  # Wavelength at reference pixel
    cdelt1 = header.get('CDELT1')  # Wavelength increment per pixel
    crpix1 = header.get('CRPIX1')  # Reference pixel (usually 1)

    # Pixel indices
    n_pix = ccddata.data.shape[-1]  # last axis is usually spectral
    pixel_indices = np.arange(n_pix)

    # Compute wavelength grid (in Angstroms, typically)
    wavelength_axis = crval1 + (pixel_indices + 1 - crpix1) * cdelt1
    
    return wavelength_axis

def heliocentric_correction(target, dateobs, ccddata, ra=None, dec=None, show_plot=True, wave_range=(4650,4720), figsize=(12,4), output=None):
    
    # Compute wavelength grid (in Angstroms, typically)
    wavelength_axis = get_wavelenght_axis(ccddata)
    
    # set OPD location
    opd = EarthLocation.from_geodetic(lat=-22.5344*u.deg, lon=-45.5825*u.deg, height=1864*u.m)
    if ra is None or dec is None:
        ra, dec = get_coordinates_from_simbad(target, show_messages=False)
    
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=Time(dateobs), location=opd)
    vhelio = heliocorr.to(u.km/u.s)
    
    wavelength_axis_corrected = apply_heliocentric_correction(wavelength_axis, vhelio.value)
    
    ccddata_corrected = ccddata.copy()
    
    ccddata_corrected.header['CRVAL1'] = wavelength_axis_corrected[0]
    ccddata_corrected.header['CDELT1'] = wavelength_axis_corrected[1]-wavelength_axis_corrected[0]
    ccddata_corrected.header['CRPIX1'] = 1
    ccddata_corrected.header['VHELIO'] = vhelio.value
    ccddata_corrected.header.comments['VHELIO'] = 'Heliocentric velocity used for correcting the spectrum (in km/s)'
    
    if show_plot:
        plt.figure(figsize=figsize)
        plt.plot(wavelength_axis,                     ccddata.data, label='No correction')
        plt.plot(wavelength_axis_corrected, ccddata_corrected.data, label=r'V$_{helio}$='+f'{vhelio:.2f}')
        
        xlim = wave_range
        xlw = (wavelength_axis > xlim[0]) * (wavelength_axis < xlim[1])
        y_sci = ccddata.data[xlw]
        ylim = (np.nanmin(y_sci),np.nanmax(y_sci))
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.title('SOURCE='+target+', DATE-OBS='+dateobs+', SITE=OPD')
        plt.ylabel(r'Flux (ADU)')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    return ccddata_corrected, vhelio.value

def save_spectrum_to_fits(ccddata, output):
    """
    Save 1D spectrum with wavelength axis as a FITS file.

    Parameters:
        filename : str
            Output FITS filename.
        ccddata (CCDData): The extracted spectrum in CCDData format, contaning the wavelength solution in the header.
    """
    
    # Create primary HDU
    hdu = fits.PrimaryHDU(data=ccddata.data, header=ccddata.header)
    hdu.writeto(output, overwrite=True)
    print(f"Saved: {output}")
