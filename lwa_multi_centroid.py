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
MAX_CENTROIDS_PER_IMAGE = 10
CENTROID_SIGMA = 6

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

def quadmax( im, sigma_x,sigma_y,xx,yy ):
    #sigma in pixels
    i,j = np.unravel_index( im.argmax(), im.shape )
    brightness = im[i,j]

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

    if i_ < ii-1 or i_ > ii+1:
        print ('wtf, polyfit borked' )
        return None,None,None,None
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

    if j_ < jj-1 or j_ > jj+1:
        print ('wtf, polyfit borked' )
        return None,None,None,None
    #last step is to evaluate the parabolic fit to get the max brightness
    aj = p[0]*j_**2 + p[1]*j_ + p[2]  
    
    # x,y = np.meshgrid( np.arange( im.shape[0]), np.arange(im.shape[1]) )
    peak = im.max() * np.exp( -(xx-j)**2/(2*sigma_x**2) -(yy-i)**2/(2*sigma_y**2) )

    return i_,j_, brightness, im-peak

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
    
    if hasattr( settings, 'maxcentroidsperimage' ):
        print( 'using %i has max centroids per image'%settings.maxcentroidsperimage )
        MAX_CENTROIDS_PER_IMAGE = settings.maxcentroidsperimage
    if hasattr( settings, 'centroidsigma' ):
        print( 'using %2.1f has centroid sigma criteria'%settings.centroidsigma )
        CENTROID_SIGMA = settings.centroidsigma

    # load the dirty image data   
    inputFile = h5py.File( settings.dirtypath, 'r' )

    if 'dirty' in inputFile.keys():
        frames = inputFile[ 'dirty' ]
    else:
        frames = inputFile[ 'dirty00' ]
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
    sigma_x = settings.speedoflight/maxFrequency / 111 #/1.141
    sigma_y = settings.speedoflight/maxFrequency / 89 #/1.141

    #convert to pixels
    dca = (frames.attrs['bbox'][0][1]-frames.attrs['bbox'][0][0])/frames.attrs['imagesize']
    print ( 'sigma',sigma_x,sigma_y, dca )
    sigma_x /= dca
    sigma_y /= dca
    print ( 'sigma',sigma_x,sigma_y )

    # sigma = 12

    ######
    # Main loop, loop until we run out of frames from the imager
    iFrame = 0
    skipCount = 0
    centroids = []
    while iFrame < NFrames:
        iSample = iFrame*settings.steptime + settings.startsample
        tSample = iSample/settings.samplerate*1000    #in m

        # if tSample < 75: 
        #     iFrame += 1
        #     continue

        frame = frames[iFrame]

        # if frame.max() == 0:
        #     print (iFrame, '0 amp')
        #     iFrame += 1
        #     skipCount += 1
        #     if skipCount > 20: break
        #     continue

        skipCount = 0

        im = frame.copy()   #we copy the frame so we can remove stuff from it
        thresh = CENTROID_SIGMA*np.std(  im )
        brightness = 0
        x,y = np.meshgrid( np.arange( im.shape[0]), np.arange(im.shape[1]) )

        #get the centroid location of the peak brightness of this frame (in indices)
        #r is the mean residual amplitude
        # i,j,r = centroid( frame, sigma=sigma )
        iCentroids = 0
        cbmax = 0
        while iCentroids < MAX_CENTROIDS_PER_IMAGE and im.max() > frame.max()/30:
            iCentroids += 1

            i,j,brightness, im = quadmax( im, sigma_x,sigma_y, x, y )
            #did we get a solution?
            if i is None: break

            #calculate residual and see if this is any good
            r = np.std(im)
            print( '%6i, %6i, %6i, %3.1f'% (iFrame, frame.max(), brightness, brightness/r) )
            #is the result still in specification?
            if brightness < CENTROID_SIGMA*r: break

            #convert to cosine projection
            ca,cb = index2cosab( i,j, frames )
            if ca**2 + cb**2 > 1:
                #we're outside the horizon
                print ('invalid centroid')
                break
            
            if cb > cbmax:
                cbmax = cb
            centroids.append( [tSample, ca, cb, brightness, r, iCentroids] )

        # if cbmax > .440:
        #     sys.exit()
        # if iCentroids > 3:
        #     print ('%i, %i'%(iSample,iCentroids))
        # if tSample > 14.0014648 and brightness == 0:
        #     sys.exit()
        # if iCentroids >= 3:
        #     if abs(dx-0.034) < 0.01:
        #         print ('***', iSample, iCentroids)
        #         sys.exit()
            
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
        

        if iFrame%1000 == 0 and len(centroids) > 0:
            if brightness is None: brightness = 0
            print ( '%1.7f %10i'%(tSample, brightness) )
            plt.cla()
            d = np.array( centroids )
            im = np.histogram2d( d[:,1], d[:,2], weights=d[:,3], bins=1000, range=[[-1,1],[-1,1]] )
            plt.imshow( im[0].T**.25, origin='lower', extent=[-1,1,-1,1]  )
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

