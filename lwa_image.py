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

####
# GLOBALS
CONFIG_PATH = 'lwa_image.cfg'

########
#do a lot of fancy stuff with the config, so that it's easy to change antenna configurations
class Settings( object ):
    """a class to store settings information
    """
    def __init__(self):
        # set some not so crazy defaults
        self.samplerate = 200000000.
        self.bandwidth  = [30000000,80000000]
        self.inputpath  = None
        self.startsample= 0

        #use all antennas, with X polarity
        self.antennas = { 'stands': range(1,257), 'polarity':0 }

def read_config( configPath=CONFIG_PATH ):
    conf = configparser.ConfigParser()
    conf.read( configPath )
    
    settings = Settings()

    ###
    # pull everything from the 'constants' section
    # these will go directly into the settings namespace
    if 'Imager' in conf.sections():
        for key, val in conf.items( 'Imager' ):
            # literal eval is a safe way of turning strings into numbers and such
            # (no little Bobby tables allowed)
            val = ast.literal_eval( val )
            # use numpy arrays when possible
            if isinstance( val, list ):
                val = np.array( val ).astype( 'float32' )
            setattr( settings, key, val )
    
    if 'Renderer' in conf.sections():
        renderer = {}
        for key, val in conf.items( 'Renderer' ):
            # literal eval is a safe way of turning strings into numbers and such
            # (no little Bobby tables allowed)
            val = ast.literal_eval( val )
            # use numpy arrays when possible
            if isinstance( val, list ):
                val = np.array( val ).astype( 'float32' )
            renderer[key] = val
        settings.renderer = renderer

    if 'Antennas' in conf.sections():
        antennas = {}
        for key, val in conf.items( 'Antennas' ):
            # literal eval is a safe way of turning strings into numbers and such
            # (no little Bobby tables allowed)
            val = ast.literal_eval( val )
            # special cases
            if key == 'stands' and val == 'all':
                val = range( 1,257 )
            # no nparrays for the antennas, these are just lists
            antennas[key] = val
        settings.antennas = antennas

    # return the settings object
    return settings

def fpad(X,M):
    """fpad(X,M)
    Frequency 0 pads X to be M*len(X) long,
    Used for fft based interpolation
    input - 
        X	-	fft of a signal, X(f)
        M	-	factor to interpolate by
    output -
        padded output(f) 
    """

    if M  <= 1:
        return X
    N = len(X)
     
    ###
    # create the output array
    output = np.zeros( N*M , dtype='complex')
    #the first N/2 samples
    output[:N//2] = X[:N//2]*M
    output[N//2]  = X[N//2]/2*M
    #the last N/2 samples
    output[-N//2] = X[N//2]/2*M
    output[-N//2+1:] = X[N//2+1:]*M
     
    return output

#########
# This is the main function block.  Will only run if program called from command line
#  (as opposed to imported)
if __name__ == '__main__':

    # load the configuration.  This can be passed as an option, so 
    # check the command line first
    if len( sys.argv ) > 1:
        # a file was given at the command line, we'll use that
        configPath = sys.argv[-1]
    else:
        # see if there's something in config
        configPath = CONFIG_PATH
    # read the config and get the settings
    settings = read_config(configPath=configPath)


    # we have an input file right?
    if not os.path.exists( settings.inputpath ):
        raise ValueError( "%s is not a valid input file"%settings.inputfile )

    # read the input file
    inputFile = h5py.File( settings.inputpath, 'r' )

    # collect all the data to be processed
    timeSeriesDsets = []
    for stand in settings.antennas['stands']:
        dsetKey = '%i_%i'%(stand, settings.antennas['polarity'])
        try:
            timeSeriesDsets.append( inputFile[dsetKey] )
        except:
            print( 'ERROR - could not access timeseries for stand %i, pol %i'%(stand,settings.antennas['polarity'] ))
            sys.exit(1)


    # where do we stop processing data?
    if settings.stopsample < 1:
        maxSample = 0
        for dset in timeSeriesDsets:
            N = dset.shape[0] - dset.attrs['integerCableDelay']
            if N > maxSample:
                maxSample = N
        settings.stopsample = maxSample

    ######
    # Calculate some array parameters which won't change during the flash
    print ('**** ****')
    ###
    # Some shorthand
    I = settings.inttime        #number of samples per integration window
    P = settings.interpolation	#interpolation for the xcorr
    M = len( timeSeriesDsets )  #the number of antennas

    ###
    # next up is the array geometry
    ang  = np.zeros( [M*(M-1)//2, 2], dtype='float32' )     #baseline angles
    dls  = np.zeros( M*(M-1)//2, dtype='float32' )          #store the delays in an array
    bls  = np.zeros( M*(M-1)//2, dtype='float32' )          #sotre the baselines in an array
    # Loop over antenna pairs
    k = 0   #location in the arrays above
    for i in range(M):
        
        iX = timeSeriesDsets[i].attrs['x']
        iY = timeSeriesDsets[i].attrs['y']

        for j in range(i+1,M):
            # edge case that shouldn't happen
            if i==j:
                continue

            jX = timeSeriesDsets[j].attrs['x']
            jY = timeSeriesDsets[j].attrs['y']


            # what's the baseline between these two antennas
            # this ignores distance in z direction
            bls[k] = np.sqrt( (iX-jX)**2 + (iY-jY)**2 )
            
            # what's the angle between the antennas?
            # this does not include z contribution
            ang[k][0] = (iX-jX)/bls[k]
            ang[k][1] = (iY-jY)/bls[k]

            # what's the delay difference
            # we've actually handled all the cable delays, and everything should be aligned
            dls[k] = 0

            k += 1
    ###
    # time weighting, to weight solution towards the center of the window
    Wt_raise = 0.16	#for hamming, Wt_raise = 0.08
    Wt = (1-np.cos( 2*np.pi*np.arange(I)/I ))*(1-Wt_raise)*.5+Wt_raise
    ###
    # Finally, the band filter for the data
    # This is a super simple tophat filter
    f = np.fft.fftfreq( 2*I )*settings.samplerate
    W = np.zeros( 2*I, dtype='int' )
    W[ (abs(f)>=settings.bandwidth[0])&(abs(f)<=settings.bandwidth[1]) ] += 1

    print ('Imaging with %i antennas'%M)
    print ('Found %i antennas, with maximum baseline %0.2f m'%(M, bls.max()))

    print ('**** ****')
    print ('Initializing Output')
    NFrames = (settings.stopsample-settings.startsample)//settings.steptime
    NImage  = settings.imagesize
    ###
    # I've tried a couple of data types, float16 saturates, as does int16
    output = np.memmap( settings.outputpath, mode='w+',  dtype='float32', shape=(NFrames,NImage,NImage) )
    # how big is the output? (hint, big)
    s = NFrames*NImage*NImage*2/1024/1024
    print ('Creating %s, sized %i MB'%(settings.outputpath, s) )

    ######
    # main loop
    print ('**** ****')
    print ('Main Loop')
    iSample = settings.startsample
    iFrame  = 0
    while iSample + settings.steptime < settings.stopsample:

        ###
        # Get Data
        data = []
        dMax = 0
        for dset in timeSeriesDsets:
            offset = dset.attrs['integerCableDelay']
            sampleGain = dset.attrs['sampleGain']*1000
            d = dset[ iSample+offset:iSample+offset+settings.inttime ]
            # don't forget to apply time weighting
            data.append( d*Wt*sampleGain )
            amp = data[-1].max() - data[-1].min()
            if amp > dMax:
                dMax = amp
        
        ###
        # Correlate
        xcs  = np.zeros( [M*(M-1)//2, I*P*2], dtype='float32' ) #store the xcross correlations in an array
        # loop over antenna pairs
        k = 0   #location in xcs
        for i in range(M):
            # get the fft of the i antenna
            ffti = np.fft.fft( data[i], 2*I )
            # whiten?
            if settings.whiten:
                # what is the mean rms amplitude of the current spectra?
                p = abs(ffti).sum().real
                # normalize the FFT (whiten) with some scaling to keep 
                # the image amplitude about the same
                ffti = ffti/abs( ffti )*p/len(ffti)

            for j in range(i+1,M):
                # edge case that shouldn't come up if loops loop right
                if i == j:
                    continue
                # get the fft of the j antenna
                fftj = np.fft.fft( data[j], 2*I )
                # whiten?
                if settings.whiten:
                    # what is the mean rms amplitude of the current spectra?
                    p = abs(fftj).sum().real
                    # normalize the FFT (whiten) with some scaling to keep 
                    # the image amplitude about the same
                    fftj = fftj/abs( fftj )*p/len(fftj)

                # compute the cross correlation
                # fpad does the interpolation
                # we toss the imag part, which should just be rounding error
                xcs[k] = np.fft.ifft( fpad( ffti*W*fftj.conj(), P ) ).real
                k += 1

        ###
        # Image
        im = imager.image( xcs, bls, dls, ang, 
            N=settings.imagesize, fs=settings.samplerate/1e6*P,
            bbox=settings.bbox, C=settings.speedoflight/1e6 )

        ###
        # Save to Output
        output[iFrame] = im

        # Some output printing, so that I know something is happening
        print( '  %10i %1.6f %i %0.1f'%(iSample, iSample/settings.samplerate, dMax, im.max()/5))

        # increment counters,
        iFrame += 1
        iSample += settings.steptime

