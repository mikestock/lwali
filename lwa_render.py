###
# Python 2/3 compatibility stuff
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt #plotting
import numpy as np #math
import h5py
import  os  #general purpose
import lwa_image as lwai #config reading

figsize = 6,6

REDNERMAX = True
INTEGRATE = True

if __name__ == '__main__':
    # load the configuration
    settings = lwai.read_config()

    # how big is the output?

       
    # load the output data
    # print ("Loading output file: %s, shape (%i,%i,%i)"%(settings.outputpath,NFrames,NImage,NImage))
    inputFile = h5py.File( settings.outputpath, 'r' )
    frames = inputFile[ 'dirty' ]
    print ('loaded file has shape: %s'%repr( frames.shape ) )
    #override settings we loaded from the config in perfernces for settings stored in the hdf5 file
    for key in frames.attrs.keys():
        setattr( settings, key, frames.attrs[key])
    # frames = np.memmap( settings.outputpath, mode='r',  dtype='float32', shape=(NFrames,NImage,NImage) )

    NFrames = (settings.stopsample-settings.startsample)//settings.steptime
    NImage  = settings.imagesize

    fig = plt.figure( figsize=figsize )
    fig.subplots_adjust( top=1,bottom=0, right=1, left=0 )
    
    # the elevation lines
    th = np.linspace( 0, 2*np.pi, 100 )
    for el in range( 10,90,10 ):
        el *= np.pi/180 #convert to radians
        el = np.cos(el)  #convert to cosine projection
        plt.plot( el*np.cos(th), el*np.sin(th), 'w-', alpha=0.2, lw=1 )
    plt.plot( np.cos(th), np.sin(th), 'w-' )
    #~ # the zenith
    #~ plt.plot( [0],[0], 'w+' )
    # the azimuth lines
    for az in range( 0,180,10 ):
        az *= np.pi/180 #convert to radians
        x = [np.cos(az), np.cos(az+np.pi)]
        y = [np.sin(az), np.sin(az+np.pi)]
        plt.plot( x,y, 'w-', alpha=0.2, lw=1 )
    
    # integrations
    imInt = np.zeros( (NImage, NImage) )
    
    ######
    # Main loop, loop until we run out of frames from the imager
    i = 0
    imMax = 0
    imMin = 1e9
    imFrame = np.zeros( (NImage, NImage) )
    vmax = settings.renderer['vmax']
    vmin = settings.renderer['vmin']
    while i < NFrames:
        
        #deconvolution
        if settings.renderer['deconvolution'] == 'max':
            # simplest deconvolution, just use the max of the dirty image
            l,m = np.unravel_index( frames[i].argmax(), frames[i].shape )
            
            im = np.zeros( (NImage, NImage) )
            im[l,m] = frames[i].max()
        else:
            # the default is no deconvolution at all
            im = frames[i].astype( 'float32' )
            # I want to do some non-linear scaling, which doesn't like 
            # negative numbers
            im[ im< 0 ] = 0
        
        # add the instantaneous frame from the imager to the current frame
        imFrame += im
        
        # tracking the maximum brightness, mostly for rendering reasons
        if im.max() > imMax:
            imMax = im.max()
        if im.max() < imMin:
            imMin = im.max()
        
        # do we render the frame yet?
        if i%settings.renderer['frameintegration'] == 0:
            # this actually displays the frame
            ret = plt.imshow( np.log( imFrame.T +1 ), extent=settings.bbox.flatten(), origin='lower', 
                interpolation='None', vmin=vmin, vmax=vmax )
            
            # Add some text with the time in the corner
            t = i*settings.steptime/settings.samplerate*1000    #in ms
            txt = fig.text( 0.05,0.95, '%1.4f ms'%t, color='w' )
            
            # Make the plotting window update
            if settings.renderer['display']:
                plt.pause( 0.0001 )
                # tmp = input( 'enter' )
            #~ print (i, im.max()**imEx, imMax**imEx)
            print (i, np.log(im.max()+1), np.log(imMin+1),np.log(imMax+1) )
            
            # save the frame output?
            if settings.renderer['saveoutput']:
                outS = settings.renderer['outputdir'] + 'frame_%06i.png'%i
                fig.savefig( outS )
            
            # remove the changing stuff
            ret.remove()
            txt.remove()
        
            # now that we've plotted stuff, reset the frame (unless we don't)
            if not settings.renderer['videointegration']:
                imFrame *= 0
        
        # update the counter
        i += 1
        
if settings.renderer['saveoutput']:
    print ('To combine the png frames into a video using mencoder:')
    print ("mencoder mf://%s -mf fps=25:type=png -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=1000000 -nosound -of lavf -lavfopts format=mp4 -o output.mp4"%('frames/frame_??????.png') )
