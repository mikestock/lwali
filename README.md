# lwali
Long Wavelength Array Lightning Imager

The Long Wavelength Array (LWA) is an astronomical radio telescope with arrays located in California and New Mexico 
(maybe in more places by the time you get around to reading this).  The arrays operate at VHF frequencies between around 
10 MHz to 90 MHz, with nominally 256 antennas distributed over a 100 meter circular array.  Details of this telescope 
can be found in the usual places

Because the telescope operates in the VHF band, it can and does see radio emission from lightning.  The codes included 
in this repository are tools designed to use LWA TBF observations to image lightning flashes.  The techniques used for 
lightning imaging tend to differ quite drastically from those used for astronomical sources because the radiation source 
is not static, and the integration times are very very short.


###############################################
Dependencies (that aren't standard python things)

1. The LWA Software Library (LSL)
   This depends on the fftw libraries being installed on the machine
2. h5py (for creating and reading hdf5 files)
3. numpy (for numpy stuff)
4. matplotlib (for rendering)

###############################################
Installation

1. Copy python files into a directory
2. Build the lwa_imager libary (requires Cython)
   python3 lwa_imager_setup.py build_ext --inplace
3. ?
4. Profit 

###############################################
Operation

1. Convert .TBF files that come from the LWA observation into a time series file
The LWA stores their transient observations in the frequency domain, split into frames of 40us
Each frame contains 12 channels, each file has enough frames for 132 channels per integration time
And there are 12 files which each cover different frequency ranges.

For lightning, we'd like to get these all combined together, and then be able to slice the 
integration times in the periods shorter than 40us.  To do this, we convert the whole thing into a
time series, 1 antenna at a time.  This is a time consuming process (several hours), but you only 
have to do it once.  
python3 tbf_to_timeseries.py <input paths> <output path>

Note, the output hdf5 file will be around 42 GB is size when complete, plan accordingly

2. Image the time series data
Edit the lwa_image.cfg file with appriate parameters then run:
python3 lwa_image.py

This is a single threaded, and not very efficient imager.  Plan on this taking approximately forever
Suggest limiting the number of antennas, and using long integration times (inttime) and step times (steptime)
to speed things up until you know everything is working ok

3. Render the dirty images
Edit the lwa_image.cfg [Render] section, then run
python3 lwa_render.py

This  will either display the results, or create .pngs, or both.  
For your convenience, the mencoder command required to turn the .pngs into an .mp4 file is printed when complete
Rendering happens pretty quickly, if you don't have clean running