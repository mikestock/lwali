###
# Python 2/3 compatibility stuff
from __future__ import print_function
from __future__ import division
# the rest of the imports
import numpy as np  #numerical package
import h5py         #read hdf5 files
import configparser, ast    #both used to read the configuration
import os, sys, time    #libraries I just tend to use
# the imager is a C library that does the imaging
import lwa_imager as imager
from lwa_image import *
import matplotlib.pyplot as plt

plt.ion()

configPath = 'lwa_image.cfg'
settings = read_config(configPath=configPath)
inputFile = h5py.File( settings.timeseriespath, 'r' )

iSample = 35392000
figsize = 6,6


def clean( im, psf, iterations=10, factor=0.75 ):
    output = np.zeros( im.shape )
    N = im.shape[0]     #size of image
    M = psf.shape[0]//2 #midpoint of psf
    imMax=im.max()
    for i in range(iterations):
        l,m = np.unravel_index( im.argmax(), im.shape )
        #break condition
        # if im[l,m] < imMax/3:
        #     break
        amplitude = factor*im[l,m]
        output[l,m] += amplitude
        im -= amplitude * psf[ M-l:M-l+N, M-m:M-m+N ]
    
    return output

###
# Some shorthand
I = settings.inttime        #number of samples per integration window
P = settings.interpolation	#interpolation for the xcorr
M = len( settings.antennas['stands'] )#the number of antennas

# collect all the data to be processed, and set integer delays
timeSeriesDsets = []
for stand in settings.antennas['stands']:
    dsetKey = '%i_%i'%(stand, settings.antennas['polarity'])
    try:
        timeSeriesDsets.append( inputFile[dsetKey] )
    except:
        print( 'ERROR - could not access timeseries for stand %i, pol %i'%(stand,settings.antennas['polarity'] ))
        sys.exit(1)

loc  = np.zeros( [M,3] )
ang  = np.zeros( [M*(M-1)//2, 2], dtype='float32' )     #baseline angles
dls  = np.zeros( M*(M-1)//2, dtype='float32' )          #store the delays in an array
bls  = np.zeros( M*(M-1)//2, dtype='float32' )          #store the baselines in an array
antennaPairs = []
intdelays  = np.zeros( M, dtype='int' )          #integer sample delays for reading in the data
#loop over antennas for the integer delays based on image center
for i in range( M ):

    #get the antenna locations        
    X = timeSeriesDsets[i].attrs['x']
    Y = timeSeriesDsets[i].attrs['y']

    #calculate baseline
    B = X**2 + Y**2
    #calculate angle components
    A = X/B, Y/B

    #calculate the delay to this antenna based on image center (in nanoseconds)
    tau = (A[0]*settings.imagecenter[0]+A[1]*settings.imagecenter[1])*B/(settings.speedoflight/1e9)
    #make this an integer number of samples
    intTau = int( tau // timeSeriesDsets[i].attrs['samplePeriod'] )

    intdelays[i] = -intTau

# Loop over antenna pairs
k = 0   #location in the arrays above
for i in range(M):
    
    iX = timeSeriesDsets[i].attrs['x']
    iY = timeSeriesDsets[i].attrs['y']
    iZ = timeSeriesDsets[i].attrs['z']
    loc[i] = iX, iY, iZ 

    for j in range(i+1,M):
        # edge case that shouldn't happen
        if i==j:
            continue

        jX = timeSeriesDsets[j].attrs['x']
        jY = timeSeriesDsets[j].attrs['y']


        D = np.sqrt( (iX-jX)**2 + (iY-jY)**2 )
        if D < settings.minbaseline: continue

        # because we don't store all the baseline pairs, 
        # we need to store the ones we do
        antennaPairs.append( (i,j) )

        # what's the baseline between these two antennas
        # this ignores distance in z direction
        bls[k] = D
        
        # what's the angle between the antennas?
        # this does not include z contribution
        ang[k][0] = (iX-jX)/bls[k]
        ang[k][1] = (iY-jY)/bls[k]

        # what's the delay difference
        # We've already handled the cable delays in the tbf conversion step
        # but, we haven't handled the beam stearing part
        tau = ( intdelays[i] - intdelays[j] ) * ( timeSeriesDsets[j].attrs['samplePeriod'] )
        dls[k] = -tau 

        k += 1

# Trim the imaging arrays
ang = ang[:k]
bls = bls[:k]
dls = dls[:k]

###
# time weighting, to weight solution towards the center of the window
Wt_raise = 0.16	#for hamming, Wt_raise = 0.08
Wt = (1-np.cos( 2*np.pi*np.arange(I)/I ))*(1-Wt_raise)*.5+Wt_raise
###
# Finally, the band filter for the data
# This is a super simple tophat filter
f = np.fft.fftfreq( 2*I )*settings.samplerate
W = np.zeros( 2*I, dtype='int' )
for bw in settings.bandwidth:
    W[ (abs(f)>=bw[0])&(abs(f)<=bw[1]) ] += 1


###
# Get Data
data = []
dMax = 0
offsets = []
for k in range( len( timeSeriesDsets) ):
    dset = timeSeriesDsets[k]
    offset = dset.attrs['integerCableDelay'] + intdelays[k]
    offsets.append( offset )
    #TODO - What is this factor of 1000 for?
    sampleGain = dset.attrs['sampleGain']*1000
    d = dset[ iSample+offset:iSample+offset+settings.inttime ]
    # don't forget to apply time weighting
    data.append( d*sampleGain )
    amp = data[-1].max() - data[-1].min()
    if amp > dMax:
        dMax = amp

###
# Correlate
xcs  = np.zeros( [len(antennaPairs), I*P*2], dtype='float32' ) #store the xcross correlations in an array

ffts = np.zeros( (M,2*I), dtype='complex64' )
spec = np.zeros( 2*I, dtype='float32' )
for i in range(M):
    ffti = np.fft.fft( data[i], 2*I )
    if settings.whiten:
        # TODO - not clear that this will works, since some of the 
        #        frequency bins (should) have 0 power in them.
        # what is the mean rms amplitude of the current spectra?
        p = abs(ffti).sum().real/len(ffti)
        # normalize the FFT (whiten) with some scaling to keep 
        # the image amplitude about the same
        # W is used here to keep data out of the passband at 0
        ffti *= p*W / abs( ffti )
    ffts[i] = ffti
    spec += abs( ffti ) / M

# loop over antenna pairs
k = 0   #location in xcs
for i,j in antennaPairs:

    # edge case that shouldn't come up if loops loop right
    if i == j:
        continue

    # compute the cross correlation
    # fpad does the interpolation
    # we toss the imag part, which should just be rounding error
    xcs[k] = np.fft.ifft( fpad( ffts[i]*W*ffts[j].conj(), P ) ).real
    k += 1

###
# Image
sys.exit()
im = imager.pimage( xcs, bls, dls, ang, 
    N=settings.imagesize, fs=settings.samplerate/1e6*P,
    bbox=settings.bbox, C=settings.speedoflight/1e6 )

im = imager.image( xcs, bls, dls, ang, 
    N=settings.imagesize, fs=settings.samplerate/1e6*P,
    bbox=settings.bbox, C=settings.speedoflight/1e6 )

fig = plt.figure( figsize=figsize )
fig.subplots_adjust( top=1,bottom=0, right=1, left=0 )

if settings.renderer['deconvolution'].lower() != 'none':
    txtcolor = 'k'
else:
    txtcolor = 'k'

# the elevation lines
th = np.linspace( 0, 2*np.pi, 100 )
for el in range( 10,90,10 ):
    el *= np.pi/180 #convert to radians
    el = np.cos(el)  #convert to cosine projection
    plt.plot( el*np.cos(th), el*np.sin(th), txtcolor+'-', alpha=0.2, lw=1 )
plt.plot( np.cos(th), np.sin(th), txtcolor+'-' )
#~ # the zenith
#~ plt.plot( [0],[0], 'w+' )
# the azimuth lines
for az in range( 0,180,10 ):
    az *= np.pi/180 #convert to radians
    x = [np.cos(az), np.cos(az+np.pi)]
    y = [np.sin(az), np.sin(az+np.pi)]
    plt.plot( x,y, txtcolor+'-', alpha=0.2, lw=1 )

#linearize the max
mx = im.max()
ret = plt.imshow( im.T, extent=settings.bbox.flatten(), origin='lower', 
    interpolation='None', vmax=mx, vmin=-mx, cmap='seismic' )     



###
# Deconvolution
xcs  = np.zeros( [bls.shape[0], I*P*2], dtype='float32' ) #store the xcross correlations in an array
for k in range( bls.shape[0] ):
    xcs[k] = np.fft.ifft( fpad( W**2, P ) ).real

# the psf is approxiamtely 2x as large as the dirty image, 
# so that we can shift it around and there's never edge artifacts
bbox = settings.bbox.copy()
bbox[0] -= bbox[0].mean()
bbox[1] -= bbox[1].mean()
pixels = settings.imagesize
bbox *= (2*pixels-1)/pixels
pixels = 2*pixels-1

# finding the psf is done the same way we did the imaging
psf = imager.pimage( xcs, bls, np.zeros( dls.shape, dtype='float32'), ang, 
    N=pixels, fs=settings.samplerate/1e6*P,
    bbox=bbox, C=settings.speedoflight/1e6 )

# normalize
psf/=psf.max()