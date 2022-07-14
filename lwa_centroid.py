###
# Python 2/3 compatibility stuff
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt #plotting
import numpy as np #math
import h5py
import  os  #general purpose
import lwa_image as lwai #config reading
import lwa_imager as imager
import sys

plt.ion()

####
# GLOBALS
CONFIG_PATH = 'lwa_image.cfg'

def centroid( im, sigma=None ):
    im = im.copy()

    #sigma in pixels
    i,j = np.unravel_index( im.argmax(), im.shape )
    x,y = np.meshgrid( np.arange( im.shape[0]), np.arange(im.shape[1]) )

    #do we have a sigma to work with, this should be the angular resolution
    #of the image
    if sigma is None:
        ii = i
        while im[ii,j] >0:
            ii += 1
            if ii >= im.shape[0]: break
        jj = j
        while im[i,jj] >0:
            jj += 1
            if jj >= im.shape[1]: break
        #this is the appoximate radius of the peak
        #whcih seems like a good sigma
        sigma = (jj-j + ii-i) / 2
        #set a lower bounds for sigma, we don't want it to not average pixels next to the peak
        if sigma <2: sigma=2


    im[im<0] = 0

    #this is a modified guassian weighting function
    exp = 4 #if this is 2, the weightning function is exactly guassian
    W = np.exp( -( abs(x-j)**exp + abs(y-i)**exp )/2/sigma**exp )

    #the centroid, based on the modified guassian weightning
    i_bar,j_bar = (W*im*y).sum()/(W*im).sum(), (W*im*x).sum()/(W*im).sum() 

    #what is left after the centroid
    resid = ((1-W)*abs(im)).sum()/((1-W).sum())

    return i_bar, j_bar, resid

def index2cosab( i,j, frames ):
    N    = frames.attrs['imagesize']
    bbox = frames.attrs['bbox']
    cosa = (bbox[0,1]-bbox[0,0])*(i+.5)/N+bbox[0,0]
    cosb = (bbox[1,1]-bbox[1,0])*(j+.5)/N+bbox[1,0]
    return cosa, cosb

if __name__ == '__main__':

    # load the configuration.  This can be passed as an option, so 
    # check the command line first
    if len( sys.argv ) > 1:
        # a file was given at the command line, we'll use that
        configPath = sys.argv[-1]
    else:
        # see if there's something in config
        configPath = CONFIG_PATH

    # load the configuration
    settings = lwai.read_config(configPath)

    # how big is the output?


    # load the output data   
    inputFile = h5py.File( settings.dirtypath, 'r' )

    frames = inputFile[ 'dirty' ]
    print ('loaded file has shape: %s'%repr( frames.shape ) )
    #override settings we loaded from the config in perfernces for settings stored in the hdf5 file
    for key in inputFile.attrs.keys():
        setattr( settings, key, inputFile.attrs[key])
    for key in frames.attrs.keys():
        setattr( settings, key, frames.attrs[key])

    NFrames = (settings.stopsample-settings.startsample)//settings.steptime
    NImage  = settings.imagesize
    print (NFrames, NImage)

    fig = plt.figure( figsize=settings.renderer['figsize'] )
    fig.subplots_adjust( top=1,bottom=0, right=1, left=0 )
    
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

    ###
    # initialize the output
    # this overwrites the output file
    outputFile = h5py.File( settings.centroidpath, mode='w' )
    outputDset = outputFile.create_dataset( 'centroids', shape=(NFrames,5), dtype='float32')

    ######
    # Main loop, loop until we run out of frames from the imager
    iFrame = 0
    
    while iFrame < NFrames:
        iSample = iFrame*settings.steptime + settings.startsample
        tSample = iSample/settings.samplerate*1000    #in ms

        frame = frames[iFrame]

        if frame.max() == 0:
            iFrame += 1
            continue

        #get the centroid location of the peak brightness of this frame (in indices)
        #r is the mean residual amplitude
        i,j,r = centroid( frame )
        ca,cb = index2cosab( i,j, frames )
        #get the amplitude of this point.  We could interpolate this, but I'm not going to bother
        brightness = frame.max() 

        # mx = abs(frame).max()
        # ret = plt.imshow( frame.T, extent=settings.bbox.flatten(), origin='lower', 
        #     interpolation='None', vmax=mx, vmin=-mx, cmap='seismic' )

        # ret2, = plt.plot( ca,cb, 'kx' )

        # plt.pause( .1 )

        # if mx > 1000:
        #     input( 'enter')

        # ret.remove()
        # ret2.remove()


        #set the output
        outputDset[ iFrame ] = tSample, ca, cb, brightness, r

        if iFrame%1000 == 0:
            print (tSample )
            plt.cla()
            im = np.histogram2d( outputDset[:,1], outputDset[:,2], weights=outputDset[:,3], bins=1000, range=[[-1,1],[-1,1]] )
            plt.imshow( im[0]**.25, origin='lower', extent=[-1,1,-1,1]  )
            plt.pause(.1 )


        iFrame += 1
    
    outputFile.close()

