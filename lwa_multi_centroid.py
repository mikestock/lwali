###
# Python 2/3 compatibility stuff
from __future__ import print_function
from __future__ import division
from email.errors import InvalidMultipartContentTransferEncodingDefect
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
    E = 16 #if this is 2, the weightning function is exactly guassian
    # W = np.exp( -( abs(x-j)**E + abs(y-i)**E )/2/sigma**E )
    W = sigma**E / ( (abs(x-j)**2 + abs(y-i)**2)**(E/2.0) + sigma**E )

    #the centroid, based on the modified guassian weightning
    i_bar,j_bar = (W*im*y).sum()/(W*im).sum(), (W*im*x).sum()/(W*im).sum() 

    #what is left after the centroid
    resid = ((1-W)*abs(im)).sum()/((1-W).sum())

    return i_bar, j_bar, resid

def quadmax( im ):
    #sigma in pixels
    i,j = np.unravel_index( im.argmax(), im.shape )
    
    #do the i max
    y = im[:,j]
    x = np.arange( im.shape[1] )
    i_ = i-1

    #handle some edge cases, and get a mask of 
    #the 3 points around the max
    ii = i
    if i <= 0: ii = 1
    if i >= len(y)-1: ii = len(y)-2
    m = [ii-1,ii,ii+1]

    #get a fit, 
    #we don't have to use polytit here, but it's convenient
    p = np.polyfit( x[m], y[m], 2 )
    #find the max of the fit, done by setting the dirivative to 0
    i_ = -p[1]/p[0]/2

    if not i_ > 0:
        print ('wtf, polyfit borked' )
    #last step is to evaluate the parabolic fit to get the max brightness
    ai = p[0]*i_**2 + p[1]*i_ + p[2] 


    #do the j max
    y = im[i,:]
    x = np.arange( im.shape[0] )

    jj = j
    if j <= 0: jj = 1
    if j >= len(y)-1: jj = len(y)-2
    m = [jj-1,jj,jj+1]
    p = np.polyfit( x[m], y[m], 2 )
    j_ = -p[1]/p[0]/2

    if not j_ > 0:
        print ('wtf, polyfit borked' )
    #last step is to evaluate the parabolic fit to get the max brightness
    aj = p[0]*j_**2 + p[1]*j_ + p[2]  
    
    return i_,j_, (ai+aj)/2, np.std( im )


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

    # load the dirty image data   
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


    #estimate the angular resolution of the interferometer, and how many pixels that is
    maxFrequency = 0
    for bw in settings.bandwidth:
        for f in bw:
            if f > maxFrequency:
                maxFrequency = f
    #this will be in image plane units
    #nominal array diameter is 100 meters
    sigma = settings.speedoflight/maxFrequency / 100

    #convert to pixels
    dca = (frames.attrs['bbox'][0][1]-frames.attrs['bbox'][0][0])/frames.attrs['imagesize']
    print ( 'sigma',sigma, dca )
    sigma /= dca
    print ( 'sigma',sigma )

    # sigma = 12

    ######
    # Main loop, loop until we run out of frames from the imager
    iFrame = 0
    centroids = []
    while iFrame < NFrames:
        iSample = iFrame*settings.steptime + settings.startsample
        tSample = iSample/settings.samplerate*1000    #in m

        # if tSample < 75: 
        #     iFrame += 1
        #     continue

        frame = frames[iFrame]

        if frame.max() == 0:
            iFrame += 1
            break

        im = frame.copy()   #we copy the frame so we can remove stuff from it
        thresh = 3*np.std(  frame )
        
        #get the centroid location of the peak brightness of this frame (in indices)
        #r is the mean residual amplitude
        # i,j,r = centroid( frame, sigma=sigma )
        iCentroids = 0
        while im.max() > frame.max() / 4 and im.max() > thresh and iCentroids < 5:
            iCentroids += 1
            # print (im.max(), frame.max())
            i,j,brightness, r = quadmax( im )
            ca,cb = index2cosab( i,j, frames )
            #get the amplitude of this point.  We could interpolate this, but I'm not going to bother
            brightness = frame.max() 
            
            centroids.append( [tSample, ca, cb, brightness, r] )

            #remove this centroid from the image
            x,y = np.meshgrid( np.arange( im.shape[0]), np.arange(im.shape[1]) )
            peak = im.max() * np.exp( - ( (x-j)**2 + (y-i)**2 )/(2*sigma**2) )
            im -= peak


        # if brightness > 100000: 
        #     break

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
        

        if iFrame%100 == 0:
            print ( '%1.7f %10i'%(tSample, brightness) )
            plt.cla()
            d = np.array( centroids )
            im = np.histogram2d( d[:,1], d[:,2], weights=d[:,3], bins=1000, range=[[-1,1],[-1,1]] )
            plt.imshow( im[0]**.25, origin='lower', extent=[-1,1,-1,1]  )
            plt.pause(.1 )


        iFrame += 1

    ###
    # initialize the output
    # this overwrites the output file
    outputFile = h5py.File( settings.centroidpath, mode='w' )
    outputDset = outputFile.create_dataset( 'centroids', data=np.array(centroids), dtype='float32')

    #copy over the attributes from the dirty file to the centroid file
    for k in inputFile.attrs.keys():
        v = inputFile.attrs[k]
        outputFile.attrs[k] = v
    for k in frames.attrs.keys():
        v = frames.attrs[k]
        outputDset.attrs[k] = v

    outputFile.close()

