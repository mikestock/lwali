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
from threading import Thread

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
        self.timeseriespath  = None
        self.startsample= 0

        #use all antennas, with X polarity
        self.antennas = { 'stands': range(1,257), 'polarity':0 }

class Processor( Thread) :
    def __init__( self, settings, bls, dls, ang ):
        Thread.__init__(self)
        self.settings = settings
        self.bls = bls
        self.dls = dls
        self.ang = ang
        self.queue = []
        self.running = True
    
    def run( self ):
        while True:
            if not self.running:
                return

            if len(self.queue) == 0:
                time.sleep( 0.1 )
                continue

            hdfFile = h5py.File( settings.dirtypath, 'a' )
            for i in range(10):
                if len( self.queue ) == 0: 
                    break
                iFrame, xcs, spec = self.queue.pop(0)
                #is this frame already processed?
                if settings.resume:
                    S =abs(hdfFile['dirty'][iFrame]).sum() 
                    if S > 0:
                        #yeah, this one is done
                        print ('skipping %i - %i'%(iFrame,S))
                        continue
                im = self.image( xcs )
                hdfFile['dirty'][iFrame] = im
                if spec is not None:
                    hdfFile['spec'][iFrame] = spec
                del spec
            print ('closing')
            hdfFile.close()
    
    def image( self, xcs ):
        im = imager.pimage( xcs, self.bls, self.dls, self.ang, 
            N=self.settings.imagesize, fs=self.settings.samplerate/1e6*P,
            bbox=self.settings.bbox, C=self.settings.speedoflight/1e6 )
        return im

    def add( self, iFrame, xcs, spec=None ):
        self.queue.append ( (iFrame, xcs, spec) )

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
                val = list(range( 1,257 ))
            # no nparrays for the antennas, these are just lists
            antennas[key] = val
        if 'excludestands' in antennas:
            for standnum in antennas['excludestands']:
                if standnum in antennas['stands']:
                    antennas['stands'].remove( standnum )
        print (antennas['stands'] )
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
    if not os.path.exists( settings.timeseriespath ):
        raise ValueError( "%s is not a valid input file"%settings.timeseriespath )

    # read the input file
    inputFile = h5py.File( settings.timeseriespath, 'r' )

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
    loc  = np.zeros( [M,3] )
    ang  = np.zeros( [M*(M-1)//2, 2], dtype='float32' )     #baseline angles
    dls  = np.zeros( M*(M-1)//2, dtype='float32' )          #store the delays in an array
    bls  = np.zeros( M*(M-1)//2, dtype='float32' )          #store the baselines in an array
    intdelays  = np.zeros( M, dtype='int' )          #integer sample delays for reading in the data
    antennaPairs = []
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
            if D > settings.maxbaseline: 
                print ('%i-%i = %2.1f exceeds max baseline'%(i,j,D))
                continue

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
    for low,high in settings.bandwidth:
        W[ (abs(f)>=low)&(abs(f)<=high) ] = 1

    print ('Imaging with %i antennas and %i baselines'%(M, len(antennaPairs) ) )
    print ('Found %i antennas, with maximum baseline %0.2f m'%(M, bls.max()))

    print ('**** ****')
    NFrames = (settings.stopsample-settings.startsample)//settings.steptime
    NImage  = settings.imagesize
    ###
    # I've tried a couple of data types, float16 saturates, as does int16
    if settings.resume and os.path.exists( settings.dirtypath ):

        print ('Appending to Output')
        outputFile = h5py.File( settings.dirtypath, mode='a' )
        outputDset = outputFile['dirty']
        # specDset   = outputFile['spec']
    else:
        print ('Initializing Output')
        outputFile = h5py.File( settings.dirtypath, mode='w' )
        outputDset = outputFile.create_dataset( 'dirty', shape=(NFrames,NImage,NImage), dtype='float32')
        #store settings information in here
        outputFile.attrs['samplerate']  = settings.samplerate
        outputFile.attrs['bandwidth']   = settings.bandwidth
        outputFile.attrs['startsample'] = settings.startsample
        outputFile.attrs['stopsample']  = settings.stopsample
        outputFile.attrs['inttime']     = settings.inttime
        outputFile.attrs['steptime']    = settings.steptime
        outputFile.attrs['interpolation'] = settings.interpolation
        outputFile.attrs['whiten']      = settings.whiten

        # if we're imaging with lots of antennas, some of these arrays get too big to fit in attrs
        # so we'll store them here instead
        outputFile.create_dataset( 'stands', data=settings.antennas['stands'] )
        outputFile.create_dataset( 'standlocs', data=loc )
        outputFile.create_dataset( 'ang', data=ang )
        outputFile.create_dataset( 'bls', data=bls )
        outputFile.create_dataset( 'dls', data=dls )

        #these are about the actual resultant image
        outputDset.attrs['imagesize']   = settings.imagesize
        outputDset.attrs['bbox']        = settings.bbox
        specDset = outputFile.create_dataset( 'spec', shape=(NFrames,2*I), dtype='float32')
    # output = np.memmap( settings.outputpath, mode='w+',  dtype='float32', shape=(NFrames,NImage,NImage) )
    # how big is the output? (hint, big)
    s = NFrames*NImage*NImage*4/1024/1024 + NFrames+2*I*4/1024/1024
    print ('Creating %s, sized %i MB'%(settings.dirtypath, s) )

    ######
    # main loop
    print ('**** ****')
    print ('Main Loop')
    iSample = settings.startsample
    iFrame  = 0
    if settings.resumesample > settings.startsample:
        #we could do this without the loop, but meh
        while iSample < settings.resumesample:
            iSample += settings.steptime
            iFrame  += 1
    if settings.resume:
        while iSample + settings.steptime < settings.stopsample:
            if abs(outputDset[iFrame]).sum() != 0:
                iFrame += 1
                iSample += settings.steptime 
                continue
            else: break
    
    outputFile.close()

    processor = Processor( settings, bls, dls, ang )
    processor.start()
    while iSample + settings.steptime < settings.stopsample:
        # if settings.resume:
        #     if outputDset[iFrame].max() > 0:
        #         iFrame += 1
        #         iSample += settings.steptime 
        #         continue

        ###
        # Get Data
        data = []
        dMax = 0
        for k in range( len( timeSeriesDsets) ):
            dset = timeSeriesDsets[k]
            offset = dset.attrs['integerCableDelay'] + intdelays[k]
            #TODO - What is this factor of 1000 for?
            sampleGain = dset.attrs['sampleGain']*1000
            d = dset[ iSample+offset:iSample+offset+settings.inttime ]
            # don't forget to apply time weighting
            data.append( d*Wt*sampleGain )
            amp = data[-1].max() - data[-1].min()
            if amp > dMax:
                dMax = amp
        
        ###
        # Correlate
        xcs  = np.zeros( [len(antennaPairs), I*P*2], dtype='float32' ) #store the xcross correlations in an array
        ffts = np.zeros( (M,2*I), dtype='complex64' )

        for i in range(M):
            ffti = np.fft.fft( data[i], 2*I )
            if settings.whiten:
                # TODO - not clear that this will works, since some of the 
                #        frequency bins (should) have 0 power in them.
                #        There is evidence that turning on whitening breaks things
                # what is the mean rms amplitude of the current spectra?
                p = abs(ffti).sum().real
                # normalize the FFT (whiten) with some scaling to keep 
                # the image amplitude about the same
                ffti = ffti/abs( ffti )*p/len(ffti)
            ffts[i] = ffti
        
        # spec = abs( ffts ).mean( axis=0 )
        spec = None


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
        # if settings.azel:
        #     im = imager.pimage_azel( xcs, bls, dls, ang, 
        #         N=settings.imagesize, fs=settings.samplerate/1e6*P,
        #         bbox=settings.bbox, C=settings.speedoflight/1e6 )
        # else:
        #     im = imager.pimage( xcs, bls, dls, ang, 
        #         N=settings.imagesize, fs=settings.samplerate/1e6*P,
        #         bbox=settings.bbox, C=settings.speedoflight/1e6 )

        while len( processor.queue ) > 10:
            time.sleep( 0.1 )
        processor.add( iFrame, xcs, spec)

        ###
        # Save to Output
        # outputDset[iFrame] = im
        # specDset[iFrame] = spec  

        # Some output printing, so that I know something is happening
        print( '  %10i %10i %1.6f %i'%(iSample, iFrame, iSample/settings.samplerate*1000, dMax ))

        # increment counters,
        iFrame += 1
        iSample += settings.steptime
