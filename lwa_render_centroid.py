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

    im = np.histogram2d( centroids[:,1], centroids[:,2], weights=centroids[:,3], bins=1000, range=[[-1,1],[-1,1]] )
    plt.imshow( im[0]**.25, origin='lower', extent=[-1,1,-1,1], vmin=0, cmap=cmap  )
    plt.pause(.1 )