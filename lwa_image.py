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
channel_1_offset = 0  #ns - This is the timedelay offset between the X and Y polarizations

########
#do a lot of fancy stuff with the config, so that it's easy to change antenna configurations
class Settings( object ):
    """a class to store settings information
    """
    def __init__(self):

        ###
        # set some not so crazy defaults
        # Paths
        self.timeseriespath  = None
        self.dirtypath       = None
        self.centroidpath    = None
        self.resume          = False
        #physical constants
        self.samplerate   = 204800000.
        self.speedoflight = 290798684
        self.bandwidth    = [[30000000,80000000]]
        #imaging constraints
        self.startsample  = 0
        self.stopsample   = -1
        self.resumesample = 0
        self.inttime      = 200
        self.steptime     = 2000
        self.interpolation= 2
        self.imagesize    = 301
        self.imagecenter  = (0,0)
        self.bbox         = [[-1.1,1.1],[-1.1,1.1]]
        self.azel         = False
        self.whiten       = False
        self.minbaseline  = 1
        self.maxbaseline  = 1000
        #deconvolution
        self.maxcentroidsperimage = 10

        #antennas
        self.antennas = {}
        self.antennas['stands'] = [1, 210, 214, 73, 249, 18, 116, 113, 255, 6, 162, 74, 242, 28, 78, 160, 211, 24, 250, 76, 77, 209, 4, 241, 25, 163, 202, 14, 245, 30, 207, 20, 115, 240, 27, 29, 246, 111, 212, 5, 203, 72, 243, 161, 75, 251, 2, 112, 254, 79, 23, 164, 239, 13, 117, 208, 3, 159, 218, 26, 213, 69, 81, 253, 10, 71, 215, 31, 201, 110, 248, 34, 66, 247, 206, 119, 109, 9, 252, 33, 216, 70, 224, 12, 204, 80, 114, 165, 21, 238, 219, 157, 118, 7, 197, 120, 22, 223, 65, 235, 32, 237, 15, 228, 37, 107, 217, 205, 11, 168, 68, 82, 156, 232, 38, 236, 16, 166, 67, 35, 108, 122, 227, 61, 121, 200, 58, 231, 43, 220, 154, 221, 155, 42, 233, 49, 234, 83, 64, 167, 196, 104, 169, 39, 226, 52, 60, 194, 36, 199, 172, 48, 153, 41, 222, 106, 123, 152, 124, 63, 230, 57, 225, 198, 84, 229, 62, 103, 170, 47, 40, 190, 51, 175, 195, 59, 44, 125, 151, 56, 181, 193, 46, 105, 176, 127, 85, 185, 100, 101, 149, 129, 148, 173, 180, 53, 86, 189, 87, 192, 55, 126, 54, 150, 45, 191, 177, 50, 184, 128, 102, 99, 88, 146, 174, 188, 93, 186, 89, 147, 132, 187, 90, 98, 182, 178, 95, 130, 145, 143, 135, 92, 183, 144, 131, 97, 179, 140, 91, 133, 96, 142, 136, 139, 134, 141, 94, 138, 137]
        self.antennas['excludestands'] = [256]
        self.antennas['[polarization'] = 0

        #renderer
        self.renderer = {}
        self.renderer['figsize']            = 6,4
        self.renderer['figdpi']             = 100
        self.renderer['bbox']               = [[-1,1],[-.44,.9]]
        self.renderer['startrender']        = 0
        self.renderer['stoprender']         = -1
        self.renderer['frameintegration']   = 100
        self.renderer['plotcentroids']      = True
        self.renderer['centroidresolution'] = 500
        self.renderer['videointegration']   = True
        self.renderer['display']            = True
        self.renderer['stepwise']           = False
        self.renderer['sampletime']         = False
        self.renderer['saveoutput']         = True
        self.renderer['outputdir']          = 'frames/'
        self.renderer['vmax']               = 5
        self.renderer['vmin']               = 0
        self.renderer['sparkle']            = True
        self.renderer['sparklemax']         = .5
        self.renderer['sparklepersist']     = .1
        self.renderer['sparkleres']         = 250
        self.renderer['sparklecmap']        = 'cool'


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
        
        if not 'calibration' in antennas:
            antennas['calibration'] = {}
        if antennas['calibration'] is None:
            antennas['calibration'] = {}
        for i in range( 1, 257 ):
            if not i in antennas['calibration']:
                antennas['calibration'][i] = [0,0,1,1]
            antennas['calibration'][i][1] += channel_1_offset

        #special case, polarization
        if 'polarity' in antennas:
            # first, fix my old typo, for backwards compatibility
            val = antennas['polarity']
            antennas.remove( 'polarity' )
            antennas['polarization'] = val
        #polarization of 0 is XX, 1 is YY, 2 is XY, 3 is YX
        #make these conform, and convert to tuples
        if not 'polarization' in antennas:
            antennas['polarization'] = 0,0
        elif antennas['polarization'] == 0: antennas['polarization'] = 0,0
        elif antennas['polarization'] == 1: antennas['polarization'] = 1,1
        elif antennas['polarization'] == 2: antennas['polarization'] = 0,1
        elif antennas['polarization'] == 3: antennas['polarization'] = 1,0

        antennas['stands'] = list( antennas['stands'] )
        if 'excludestands' in antennas:
            for stand in antennas['excludestands']:
                if stand in antennas['stands']:
                    antennas['stands'].remove( stand )

        #actually store the antenna settings in settings, seems like a good idea
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

def correlate( pair ):
    """
    This does the FFT correlation for 1 antenna pair (i,j)
    The intent is to use this in a multiprocessing map, to get them 
    all done at once
    """
    i,j = pair
    return np.fft.ifft( ffts[i][0]*W*ffts[j][1].conj() ).real

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
        if stand in settings.antennas['excludestands']:
            continue
        dsetKeyX = '%i_%i'%(stand, 0)
        dsetKeyY = '%i_%i'%(stand, 1)
        try:
            timeSeriesDsets.append( (inputFile[dsetKeyX],inputFile[dsetKeyY]) )
        except:
            print( 'ERROR - could not access timeseries for stand %i'%(stand ) )
            sys.exit(1)


    # where do we stop processing data?
    if settings.stopsample < 1:
        maxSample = 0
        for dsetX,dsetY in timeSeriesDsets:
            N = dsetX.shape[0] - dsetX.attrs['integerCableDelay']
            if N > maxSample:
                maxSample = N
        settings.stopsample = maxSample

    ######
    # Calculate some array parameters which won't change during the flash
    print ('**** ****\nPre-computing things')
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
        X = timeSeriesDsets[i][0].attrs['x']
        Y = timeSeriesDsets[i][0].attrs['y']

        #calculate baseline
        B = X**2 + Y**2
        #calculate angle components
        A = X/B, Y/B

        #calculate the delay to this antenna based on image center (in nanoseconds)
        tau = (A[0]*settings.imagecenter[0]+A[1]*settings.imagecenter[1])*B/(settings.speedoflight/1e9)
        #make this an integer number of samples
        intTau = int( tau // timeSeriesDsets[i][0].attrs['samplePeriod'] )

        intdelays[i] = -intTau

    # Loop over antenna pairs
    k = 0   #location in the arrays above
    for i in range(M):
        
        iX = timeSeriesDsets[i][0].attrs['x']
        iY = timeSeriesDsets[i][0].attrs['y']
        iZ = timeSeriesDsets[i][0].attrs['z']
        iStand = timeSeriesDsets[i][0].attrs['stand']
        loc[i] = iX, iY, iZ 

        for j in range(i+1,M):
            # edge case that shouldn't happen
            if i==j:
                continue

            jX = timeSeriesDsets[j][0].attrs['x']
            jY = timeSeriesDsets[j][0].attrs['y']
            jStand = timeSeriesDsets[j][0].attrs['stand']


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
            tau = ( intdelays[i] - intdelays[j] ) * ( timeSeriesDsets[j][0].attrs['samplePeriod'] )

            pol = settings.antennas['polarization']
            dls[k] = -tau +settings.antennas['calibration'][iStand][pol[0]]-settings.antennas['calibration'][jStand][pol[1]]
            k += 1
    
    # Trim the imaging arrays
    ang = ang[:k]
    bls = bls[:k]
    dls = dls[:k]
    antennaPairs = np.array( antennaPairs, dtype='i' )

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
        #even though this file has been run before, it may not have been done for this polarization
        for i in range( 2 ):
            for j in range(2 ):
                outputKey = 'dirty%i%i'%(i,j)
                if not outputKey in outputFile:
                    print ('creating %s'%outputKey)
                    outputDset = outputFile.create_dataset( outputKey, shape=(NFrames,NImage,NImage), dtype='float32')
                    outputDset.attrs['polarization']= settings.antennas['polarization']
                    outputDset.attrs['imagesize']   = settings.imagesize
                    outputDset.attrs['bbox']        = settings.bbox
        outputKey = 'dirty%i%i'%settings.antennas['polarization']
        outputDset = outputFile[outputKey]
        specDset   = outputFile['spec']
    else:
        print ('Initializing Output')
        outputFile = h5py.File( settings.dirtypath, mode='w' )
        for i in range( 2 ):
            for j in range(2 ):
                outputKey = 'dirty%i%i'%(i,j)
                print ('creating %s'%outputKey)
                outputDset = outputFile.create_dataset( outputKey, shape=(NFrames,NImage,NImage), dtype='float32')
                outputDset.attrs['polarization']= settings.antennas['polarization']
                outputDset.attrs['imagesize']   = settings.imagesize
                outputDset.attrs['bbox']        = settings.bbox
        outputKey = 'dirty%i%i'%settings.antennas['polarization']
        outputDset = outputFile[outputKey]
        #store settings information in here
        outputFile.attrs['samplerate']  = settings.samplerate
        outputFile.attrs['bandwidth']   = settings.bandwidth
        outputFile.attrs['startsample'] = settings.startsample
        outputFile.attrs['stopsample']  = settings.stopsample
        outputFile.attrs['inttime']     = settings.inttime
        outputFile.attrs['steptime']    = settings.steptime
        outputFile.attrs['interpolation'] = settings.interpolation
        outputFile.attrs['whiten']      = settings.whiten
        ###
        # If the number of stands is too big, these guys won't actually fit in an attribute
        # TODO - fix this
        if len( loc ) < 32:
            outputFile.attrs['stands']      = settings.antennas['stands']
            outputFile.attrs['standlocs']   = loc
            outputFile.attrs['ang']         = ang
            outputFile.attrs['bls']         = bls
            outputFile.attrs['dls']         = dls

        #these are about the actual resultant image
        outputDset.attrs['polarization']= settings.antennas['polarization']
        outputDset.attrs['imagesize']   = settings.imagesize
        outputDset.attrs['bbox']        = settings.bbox
        specDset = outputFile.create_dataset( 'spec', shape=(NFrames,2*I), dtype='float32')
    # output = np.memmap( settings.outputpath, mode='w+',  dtype='float32', shape=(NFrames,NImage,NImage) )
    # how big is the output? (hint, big)
    s = NFrames*NImage*NImage*2/1024/1024
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

    tStart = time.time()
    framesProcessed = 0
    while iSample + settings.steptime < settings.stopsample:
        if settings.resume:
            if abs(outputDset[iFrame]).max() > 0:
                iFrame += 1
                iSample += settings.steptime 
                continue
            
        ###
        # Get Data
        data = []
        dMax = 0
        for k in range( len( timeSeriesDsets) ):
            #dset0 and dset1 might be the same dset
            dset0 = timeSeriesDsets[k][ settings.antennas['polarization'][0] ]
            dset1 = timeSeriesDsets[k][ settings.antennas['polarization'][1] ]
            offset = dset0.attrs['integerCableDelay'] + intdelays[k]
            sampleGain = dset0.attrs['sampleGain']
            stand      = dset0.attrs['stand']
            d0 = dset0[ iSample+offset:iSample+offset+settings.inttime ] * Wt*sampleGain
            d1 = dset1[ iSample+offset:iSample+offset+settings.inttime ] * Wt*sampleGain
            ###
            # apply calbration gain
            pol = settings.antennas['polarization']
            d0 *= settings.antennas['calibration'][stand][ pol[0]+2 ]
            d1 *= settings.antennas['calibration'][stand][ pol[1]+2 ]
            # don't forget to apply time weighting
            data.append( [d0,d1] )
            amp = data[-1][0].max() - data[-1][0].min()
            if amp > dMax:
                dMax = amp
        
        ###
        # Correlate
        xcs  = np.zeros( [len(antennaPairs), I*P*2], dtype='float32' ) #store the xcross correlations in an array
        ffts = np.zeros( (M,2,2*I), dtype='complex128' )

        spec = np.zeros( 2*I )
        for i in range(M):
            for j in range(2):
                fftij = np.fft.fft( data[i][j], 2*I )
                if settings.whiten:
                    # TODO - not clear that this will works, since some of the 
                    #        frequency bins (should) have 0 power in them.
                    #        There is evidence that turning on whitening breaks things
                    # what is the mean rms amplitude of the current spectra?
                    p = abs(fftij).sum().real
                    # normalize the FFT (whiten) with some scaling to keep 
                    # the image amplitude about the same
                    fftij = fftij/abs( fftij )*p/len(fftij)
                spec += abs( fftij )
                ffts[i][j] = fftij
        
        #mean across stands and polarizations
        spec /= ffts.shape[0]*2

        k = 0   #location in xcs
        for i,j in antennaPairs:

            # edge case that shouldn't come up if loops loop right
            if i == j:
                continue

            # compute the cross correlation
            # fpad does the interpolation
            # the [0] and [1] select the polarization of the signals.
            # we toss the imag part, which should just be rounding error
            xcs[k] = np.fft.ifft( fpad( ffts[i][0]*W*ffts[j][1].conj(), P ) ).real
            # xcs[k] = np.fft.ifft( ffts[i][0]*W*ffts[j][1].conj() ).real
            k += 1

        ###
        # Image
        if settings.azel:
            im = imager.pimage_azel( xcs, bls, dls, ang, 
                N=settings.imagesize, fs=settings.samplerate/1e6*P,
                bbox=settings.bbox, C=settings.speedoflight/1e6 )
        else:
            im = imager.pimage( xcs, bls, dls, ang, 
                N=settings.imagesize, fs=settings.samplerate/1e6*P,
                bbox=settings.bbox, C=settings.speedoflight/1e6 )

        ###
        # Save to Output
        outputDset[iFrame] = im
        specDset[iFrame] = spec  

        # Some output printing, so that I know something is happening
        framesProcessed += 1
        print( '  %10i %1.4f %0.2f %6.3f %1.1f'%(iSample, iSample/settings.samplerate*1000, dMax, im.max()/5, (time.time()-tStart)/framesProcessed ))

        # increment counters,
        iFrame += 1
        iSample += settings.steptime

    outputFile.close()
