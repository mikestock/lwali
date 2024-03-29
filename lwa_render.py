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
from matplotlib.colors import LinearSegmentedColormap

####
# GLOBALS
CONFIG_PATH = 'lwa_image.cfg'

###
# create a colormap which is all black, and just maps transparency
cdict = { 
            'red'  : [(0,0,0),(1,0,0)],
            'green': [(0,0,0),(1,0,0)],
            'blue' : [(0,0,0),(1,0,0)],
            'alpha': [(0,0,0),(1,1,1)]
}
cmap = LinearSegmentedColormap( 'cmap', segmentdata=cdict, N=256)

###
# create a colormap that goes blue, then while, then red
cdict = { 
            'red'  : [  (0.00,0.0,0.0),
                        (0.45,0.2,0.2),
                        (0.50,1.0,1.0),
                        (0.55,1.0,1.0),
                        (1.00,0.5,0.5)],
            'green': [  (0.00,0.0,0.0),
                        (0.45,0.2,0.2),
                        (0.50,1.0,1.0),
                        (0.55,0.2,0.2),
                        (1.00,0.0,0.0)],
            'blue' : [  (0.00,0.5,0.5),
                        (0.45,1.0,1.0),
                        (0.50,1.0,1.0),
                        (0.55,0.2,0.2),
                        (1.00,0.0,0.0)],
}
cmap_rwb = LinearSegmentedColormap( 'cmap', segmentdata=cdict, N=1024)

txtcolor = 'k'  #sets the color of text, and line segments

def clean( im, psf, iterations=10, factor=0.75 ):
    output = np.zeros( im.shape )
    N = im.shape[0]     #size of image
    M = psf.shape[0]//2 #midpoint of psf
    imMax=im.max()
    for i in range(iterations):
        l,m = np.unravel_index( im.argmax(), im.shape )
        #break condition
        if im[l,m] < imMax/3:
            break
        amplitude = factor*im[l,m]
        output[l,m] += amplitude
        im -= amplitude * psf[ M-l:M-l+N, M-m:M-m+N ]
    
    return output


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

    ###
    # fix the speed of light issue seen in some early version of the imager
    fixc = 1
    if 'fixc' in settings.renderer:
        if settings.renderer['fixc']:
            fixc = 299792458./290798684

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
    
    if settings.renderer['deconvolution'].lower() in ['none', 'dirty+']:
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
    
    if settings.renderer['deconvolution'] == 'clean':
        print ( 'Generating PSF for cleaning' )
        #we need to compute the point spread function
        #this will be based on an average spectra for the flash
        I = inputFile.attrs['inttime']
        M = len( inputFile.attrs['stands'] )
        P = inputFile.attrs['interpolation']	#interpolation for the xcorr
        bls = inputFile.attrs['bls'][:]
        ang = inputFile.attrs['ang'][:]
        dls = inputFile.attrs['dls'][:]

        # this averaging here will tend to smooth out the PSF
        # maybe you could get better results if you calculated the PDF for each 
        # integration period, but that would be basically imaging twice
        spec = inputFile['spec'][:].mean( axis=0 )

        xcs  = np.zeros( [bls.shape[0], I*P*2], dtype='float32' ) #store the xcross correlations in an array
        for k in range( bls.shape[0] ):
            xcs[k] = np.fft.ifft( lwai.fpad( spec**2, P ) ).real

        # the psf is approxiamtely 2x as large as the dirty image, 
        # so that we can shift it around and there's never edge artifacts
        bbox = settings.bbox.copy()
        bbox[0] -= bbox[0].mean()
        bbox[1] -= bbox[1].mean()
        pixels = inputFile['dirty'].attrs['imagesize']
        bbox *= (2*pixels-1)/pixels
        pixels = 2*pixels-1

        # finding the psf is done the same way we did the imaging
        psf = imager.image( xcs, bls, dls, ang, 
            N=pixels, fs=inputFile.attrs['samplerate']/1e6*P,
            bbox=bbox, C=settings.speedoflight/1e6 )
        
        # normalize
        psf/=psf.max()

    if settings.renderer['plotcentroids']:
        centroidFile = h5py.File( settings.centroidpath, 'r' )
        centroids = centroidFile['centroids']

    # integrations
    imInt = np.zeros( (NImage, NImage) )
    
    ######
    # Main loop, loop until we run out of frames from the imager
    i = 0
    imMax = 0
    imMin = 1e9
    imFrame   = np.zeros( (NImage, NImage) )
    imSparkle = np.zeros( (NImage, NImage) )    #always current frame
    vmax = settings.renderer['vmax']
    vmin = settings.renderer['vmin']
    sparklemax = settings.renderer['sparklemax']
    lastCentroid = 0
    while i < NFrames:
        iSample = i*settings.steptime + settings.startsample
        if iSample < settings.renderer['startrender'] and not settings.renderer['videointegration']:
            #we haven't gotten to the section of the file we want to render
            i += 1
            continue
        if iSample > settings.renderer['stoprender'] and settings.renderer['stoprender'] > 0:
            #we're done rendering
            break
        #deconvolution
        if settings.renderer['deconvolution'] == 'max':
            # simplest deconvolution, just use the max of the dirty image
            l,m = np.unravel_index( frames[i].argmax(), frames[i].shape )
            
            im = np.zeros( (NImage, NImage) )
            im[l,m] = frames[i].max()
        elif settings.renderer['deconvolution'] == 'clean':
            factor = settings.renderer['cleaningfactor']
            iterations = settings.renderer['cleaningiterations']
            im = clean( frames[i], psf, iterations, factor)
        else:
            # the default is no deconvolution at all
            im = frames[i].astype( 'float32' )
            # I want to do some non-linear scaling, which doesn't like 
            # negative numbers
            # im[ im< 0 ] = 0
      
        
        # tracking the maximum brightness, mostly for rendering reasons
        if im.max() > imMax:
            imMax = im.max()
        if im.max() < imMin:
            imMin = im.max()

        # add the instantaneous frame from the imager to the current frame
        if settings.renderer['deconvolution'].lower() not in ['none', 'dirty+']:
            im[ np.log(im +1) < vmin ] = 0
            imFrame += im
            imSparkle += im
        else:
            imFrame += im

        if iSample < settings.renderer['startrender']:
            #we've done the video integration, so we're done now
            i += 1
            continue        

        # do we render the frame yet?
        if i%settings.renderer['frameintegration'] == 0:
            # this actually displays the frame
            if settings.renderer['deconvolution'].lower() not in ['none', 'dirty+']:
                ret = plt.imshow( np.log( imFrame.T +1 ), extent=settings.bbox.flatten(), origin='lower', 
                    interpolation='None', vmin=vmin, vmax=vmax, cmap='binary' )
            elif settings.renderer['deconvolution'].lower() == 'none':
                #linearize the max
                mx = sparklemax**4
                print( im.max(), mx )
                mx = im.max()
                ret = plt.imshow( im.T, extent=settings.bbox.flatten()*fixc, origin='lower', 
                    interpolation='None', vmax=mx, vmin=-mx, cmap=cmap_rwb )
            elif settings.renderer['deconvolution'].lower() == 'dirty+':
                # im[im<im.max()/10] = 0
                im[im<0] = 0
                im = im **.5
                #linearize the max
                mx = sparklemax
                print( 10*np.log10(frames[i].max()/5.692e-5/settings.inttime) ,im.max(), mx )
                ret = plt.imshow( im.T, extent=settings.bbox.flatten()*fixc, origin='lower', 
                    interpolation='None', vmax=mx, vmin=-mx, cmap='seismic' )

            # Sparkles
            if settings.renderer['sparkle'] and settings.renderer['deconvolution'].lower() not in ['none', 'dirty+']:
                # imSparkle[ imSparkle > 0 ] = 1
                im = np.log( imSparkle +1 )
                # imSparkle /= settings.renderer['sparklemax']
                im[ im == 0 ] = np.nan
                ret2 = plt.imshow( im.T , extent=settings.bbox.flatten()*fixc, origin='lower', 
                    interpolation='None', vmin=vmin, vmax=sparklemax, cmap=settings.renderer['sparklecmap'] )

            #Centroids
            if settings.renderer['plotcentroids']:
                tSample = iSample/settings.samplerate*1000    #in ms
                if tSample-lastCentroid > 0.05:
                    m = centroids[:,0] < tSample 
                    im2 = np.histogram2d( centroids[m,1]*fixc, centroids[m,2]*fixc, weights=centroids[m,3], bins=1000, range=[[-1,1],[-1,1]] )
                    lastCentroid = tSample
                ret3 = plt.imshow( im2[0].T**.25, origin='lower', extent=[-1,1,-1,1], vmin=0, cmap=cmap  )


            # Add some text with the time in the corner
            if not settings.renderer['sampletime']:
                t = iSample/settings.samplerate*1000    #in ms
                txt = fig.text( 0.05,0.95, '%1.4f ms'%t, color=txtcolor )
            else:
                t = iSample
                txt = fig.text( 0.05,0.95, '%i'%t, color=txtcolor )
            
            # Make the plotting window update
            if settings.renderer['display']:
                plt.pause( 0.0001 )
                if settings.renderer['stepwise']:
                    tmp = input( 'enter' )
            #~ print (i, im.max()**imEx, imMax**imEx)
            print (i, np.log(imFrame.max()+1), np.log(imMin+1), np.log(imMax+1) )

            #set the limits
            if settings.renderer['bbox'] is not None:
                plt.xlim( settings.renderer['bbox'][0] )
                plt.ylim( settings.renderer['bbox'][1] )
            #turn off axes ticks
            plt.xticks( [] )
            plt.yticks( [] )

            # save the frame output?
            if settings.renderer['saveoutput']:
                outS = settings.renderer['outputdir'] + 'frame_%06i.png'%i
                fig.savefig( outS, dpi=settings.renderer['figdpi'] )
            
            # remove the changing stuff
            ret.remove()
            txt.remove()
            if settings.renderer['sparkle'] and settings.renderer['deconvolution'].lower() not in ['none', 'dirty+']:
                ret2.remove()
            if settings.renderer['plotcentroids']:
                ret3.remove()
        
            # now that we've plotted stuff, reset the frame (unless we don't)
            if not settings.renderer['videointegration']:
                imFrame *= 0
            # imSparkle *= np.zeros( (NImage, NImage) )    #always current frame
            imSparkle *= settings.renderer['sparklepersist']
            imSparkle[ np.log( imSparkle+1) < vmin ] = 0

        # update the counter
        i += 1
        
if settings.renderer['saveoutput']:
    print ('To combine the png frames into a video using mencoder:')
    print ("mencoder mf://%s -mf fps=25:type=png -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=1000000 -nosound -of lavf -lavfopts format=mp4 -o output.mp4"%('frames/frame_??????.png') )
