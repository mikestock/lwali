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


###
# create a colormap which is all black, and just maps transparency
cdict = { 
            'red'  : [(0,0,0),(1,0,0)],
            'green': [(0,0,0),(1,0,0)],
            'blue' : [(0,0,0),(1,0,0)],
            'alpha': [(0,0,0),(1,1,1)]
}
cmap = LinearSegmentedColormap( 'cmap', segmentdata=cdict, N=256)
txtcolor = 'k'  #sets the color of text, and line segments

####
# GLOBALS
CONFIG_PATH = 'lwa_image.cfg'

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

    inputFile = h5py.File( settings.centroidpath, 'r' )
    centroids = inputFile['centroids']

    ###
    # fix the speed of light issue seen in some early version of the imager
    fixc = 0
    if 'fixc' in settings.renderer:
        if settings.renderer['fixc']:
            fixc = 299792458./290798684

    #remove things we're not supposed to render
    sTime = centroids[:,0]*1e-3*settings.samplerate
    if settings.renderer['startrender'] > 0:
        m = sTime > settings.renderer['startrender']
    else:
        m = np.ones( len(centroids), dtype='bool')
    if settings.renderer['stoprender'] > 0:
        m&= sTime < settings.renderer['stoprender']
    centroids = centroids[:][m]

    fig = plt.figure( figsize=settings.renderer['figsize'] )
    fig.subplots_adjust( top=1,bottom=0, right=1, left=0 )

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


    bgPixels  = settings.renderer['centroidresolution']
    spPixels  = settings.renderer['sparkleres']
    bbox = settings.renderer['bbox']

    dx = bbox[0][1]-bbox[0][0]
    dy = bbox[1][1]-bbox[1][0]
    if dx > dy:
        bgPixelsX = bgPixels
        spPixelsX = spPixels
        bgPixelsY = int( bgPixels*dy/dx )
        spPixelsY = int( spPixels*dy/dx )
    else:
        bgPixelsY = bgPixels
        spPixelsY = spPixels
        bgPixelsX = int( bgPixels*dx/dy )
        spPixelsX = int( spPixels*dx/dy )

    im = np.histogram2d( centroids[:,1]*fixc, centroids[:,2]*fixc, weights=centroids[:,3], bins=[bgPixelsX,bgPixelsY], range=bbox )
    plt.imshow( im[0].T**.25, origin='lower', extent=bbox.flatten(), vmin=0, cmap=cmap  )
    plt.pause(.1 )