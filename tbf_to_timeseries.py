from lsl.common import stations
from lsl.reader import tbf, errors
import numpy as np
import h5py, sys, argparse

# import matplotlib.pyplot as plt

###
# The station is information on the array built into LSL
station = stations.lwasv
# in the station is a list of antennas
antennas = station.antennas #this is a list of antenna objects
#we could loop through these antennas looking for good ones, but since they're just coming from the 
#built in library information, I'm not sure how any of them wouldn't be.  
#the list of antennas is ordered, with X pol antennas having pol=0, and Y pol antennas having pol=1

totalChannels  = 4096
frequencyRange = 40000,88000    #this is really only used for calculating cable delays
speedOfLight   = 299792458.0



if __name__ == '__main__' :
    #parse the command line
    parser = argparse.ArgumentParser(
        description='Convert a TBF file into a continuous timeseries', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument('input_paths', type=str, nargs='+',
                        help='filenames to convert')
    parser.add_argument('output_path', type=str, 
                        help='where to store the output')

    args = parser.parse_args()
    # main( args )
# def main( args ):
    outputFile = h5py.File( args.output_path, 'w' )

    #store combined spectrum stuff here
    for iStand in range(256):
        for iPol in range(2):
            print( 'Reading data for Stand %i Pol %i'%(iStand+1, iPol))
            #loop over (unordered) list of files
            combinedSpec = {}
            triggerTime  = 0
            for inputPath in args.input_paths:

                ###
                # there are 12 files per trigger, each with the same start time but different frequency ranges
                # in one file, data is stored in chunks of 12 channels each, for all 256 antennas.  These are called 'frames'
                print( 'reading %s'%inputPath )
                fh = open( inputPath, 'rb')

                # Figure out how many frames there are per observation and the number of
                # channels that are in the file
                nFramesPerObs = tbf.get_frames_per_obs(fh)
                nchannels = tbf.get_channel_count(fh)

                #read the whole damn file
                while True:
                    
                    #we're gonna shove the spectra data into here
                    fileEnd = False
                    for iChunk in range( nFramesPerObs ):
                        try:
                            cFrame = tbf.read_frame(fh)
                        except errors.EOFError:
                            fileEnd = True
                            break

                        #the data for the cFrame is stored in cFrame.payload.data
                        #it's a 12x256x2 array, where there are:
                        #12 channels  - frequency information
                        #256 stands   - this is antenna.stand.id -1
                        #2 polarities - this is antenna.pol
                        #in a lot of the example scripts Jayce reshapes this, so that you have a 
                        #12x512 array which indexes to antenna.id-1 or antenna.digitizer-1 (same thing?)

                        # specTime = cFrame.time.unix
                        specTime = int(cFrame.time.unix*1000000000)    #in ns since 1970
                        if specTime < triggerTime or triggerTime == 0:
                            triggerTime = specTime
                        if not specTime in combinedSpec:
                            combinedSpec[specTime] = np.zeros( totalChannels+1, dtype='complex64' )

                        first_chan = cFrame.header.first_chan
                        if first_chan+12 > totalChannels:
                            print( 'channel out of range' )
                            continue

                        combinedSpec[specTime][first_chan:first_chan+12] = cFrame.payload.data[:,iStand,iPol]
                    
                    if fileEnd: break

            #convert to timeseries
            print( 'Spliting frames into continuous time series')
            nSamples = len( combinedSpec ) * totalChannels*2   #we'll have this many samples in each one
            maxFrequency = totalChannels * 25000
            samplePeriod = 1000000000/(2*maxFrequency)  #in nanoseconds

            triggerTime = min( combinedSpec.keys() )
            iChunk = 0
            iTimeSeries = 0
            timeSeries  = np.zeros( [nSamples ], dtype='float32' )
            for specTime in sorted( combinedSpec.keys() ):
                expectedTime = iChunk*samplePeriod*2*totalChannels
                if abs(specTime-triggerTime- expectedTime ) > 500:
                    print( 'ERROR timing tracking out os sync!!' )
                    sys.exit(1)

                timeSeries[ iTimeSeries:iTimeSeries+2*totalChannels ] = np.fft.irfft( combinedSpec[specTime] )
                iChunk += 1
                iTimeSeries += 2*totalChannels

            #apply correction for cable delay and dispersion
            #start by finding the right antenna
            for ant in antennas:
                if ant.stand.id-1 == iStand and ant.pol == iPol:
                    break
            if not (ant.stand.id-1 == iStand and ant.pol == iPol):
                print( 'ERROR Count not find antenna for stand %i pol %i'%(iStand+1, iPol) )
                sys.exit(1)

            #TODO, confirm this isn't borked up here

            N = int(len(timeSeries)/2+1)
            freq = np.linspace( 0, totalChannels*25000, N )
            m = (freq>frequencyRange[0]*1000)&(freq<frequencyRange[1]*1000)
            #this is the cable delay, measured in sampled
            delay = ant.cable.delay( freq[m] )*1e9/samplePeriod  
            #remove the smallest integer number of samples from this delay
            integerCableDelay = int( delay.min() )
            delay -= integerCableDelay

            phaseRot = np.exp( 1j*np.pi*np.arange(N)[m]*delay/N )
            # this is written for delays in nseconds
            # phaseRot = np.exp(2j*np.pi*freq[m]*( delay-ant.stand.z/speedOfLight) )
            gain = np.sqrt( ant.cable.gain(freq[m]) )

            print ('Applying cable delays, this is 2 very large FFTs, stand by')
            spec = np.fft.rfft( timeSeries )
            spec[m] *= phaseRot/gain
            timeSeries = np.fft.irfft( spec )

            #quantize the time series so we can put this into ints and have it not be so massive giant
            sampleGain = abs(timeSeries.max())/2**15
            timeSeries = ( timeSeries/sampleGain ).astype( 'int16' )

            print ('Writing to hdf file' )
            dset = outputFile.create_dataset( '%i_%i'%(iStand+1,iPol), data=timeSeries, dtype='int16', chunks=(8000,), compression="gzip")
            dset.attrs['stand'] = iStand+1
            dset.attrs['pol'] = iPol
            dset.attrs['integerCableDelay'] = integerCableDelay
            dset.attrs['samplePeriod'] = samplePeriod
            dset.attrs['maxFrequency'] = maxFrequency
            dset.attrs['sampleGain']   = sampleGain
            dset.attrs['x'] = ant.stand.x
            dset.attrs['y'] = ant.stand.y
            dset.attrs['z'] = ant.stand.z
            dset.attrs['triggerTime'] = triggerTime

    outputFile.close()
