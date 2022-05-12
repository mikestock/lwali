# cython: boundscheck=False
# cython: cdivision=True
cimport numpy as np
import numpy as np
from libc.math cimport sin, cos, M_PI, sqrt
from libc.stdlib cimport malloc, free
from cython.parallel import prange

# cdef float quadint( float a1, float a2, float a3, float i ) nogil:
#     """
#     The same as quadint_arr, but without the arrays for 
#     more multiprocessing happy fun times

#     i should be a float between 0 and 1
#     a1
#     """

cdef float quadint( float[:,:] arr, int k, float i ):
    """quadint
    quadratic interpolation, 
    hard coded implementation
    
    Based on numerical recipes in C for poly interpolation modified 
    with the loops unwrapped and only for the quadratic case

    k is the antenna index
    i is the (floating point) lagtime index to interpolate
    """
    
    cdef int j = <int> i
    #these are to index the array
    #we generate them here so that the can wrap 
    #properly if needed
    cdef int j1 = j+1, j2 = j+2
    cdef int N = arr.shape[1]
    if j1 >= N:
        j1 -= N
    if j2 >= N:
        j2 -= N
    
    # cdef float Output = 0, a
    # #there are 3 terms
    # a = .5*(i-j-1)*(i-j-2)
    # Output += a*arr[k,j]
    # a = -1*(i-j)*(i-j-2)
    # Output += a*arr[k,j1]
    # a = .5*(i-j)*(i-j-1)
    # Output += a*arr[k,j2]
    
    cdef float Output = 0
    Output = .5*(i-j-1) * (i-j-2) * arr[k,j] +\
             -1*(i-j)   * (i-j-2) * arr[k,j1] +\
             .5*(i-j)   * (i-j-1) * arr[k,j2]

    return Output	

def image( 	np.ndarray[	float, ndim=2] xc, 
                    np.ndarray[	float, ndim=1] bl, 
                    np.ndarray[	float, ndim=1] dl,
                    np.ndarray[ float, ndim=2] A, 
                    int N=50, 
                    float fs=200,
                    np.ndarray[ float, ndim=2] bbox = np.array( [[-1.1,1.1],[-1.1,1.1]], dtype='float32' ),
                    float C=299.792458
                    ):
    
    """
    Computes the image of the sky from M baselines
    xc - cross correlations
    bl - baseline lengths in meters
    dl - extra delays (eg cable lengths) in ns
    A  - baseline orientation matrix
    N  - number of pixels for the image
    fs - sampling frequency (after interpolation)
    bbox - the edges of the image to be made, should be square
    
    image uses a projection based algorithm which is fast, to project 
    the cross correlations onto the sky.  No non-linear corrections are 
    applied (like distance to the source).  
    
    Uses quadratic interpolation to get sub-sample values of the 
    cross correlations.  This is fast, but not exact.  Results will be 
    exact if the cross correlations are heavily interpolated, at the 
    expense of computation time.
    """
    cdef float cosa, cosb, tau, dtau

    #these are used to calculate the amplitude of the pixel
    cdef float p, l
        
    #counters, there's a bunch of them
    cdef int i,j,k,m

    #int, corresponds to tau=0
    #this wouldn't be 0 if we used fftshift on the cross correlations
    cdef int  mMiddle = 0

    #the size of the grid
    cdef float dcosa = (bbox[0,1]-bbox[0,0])/N*.5
    cdef float dcosb = (bbox[1,1]-bbox[1,0])/N*.5

    #the shape of the xc array, not as python values
    cdef int xcN = <int> xc.shape[0]
    cdef int xcM = <int> xc.shape[1]

    ###
    # Generate the image array
    # can't be empty since not all values will be calculated
    cdef float pixelValue
    cdef np.ndarray[float, ndim=2] Output = np.zeros( (N,N), dtype='float32' )
    
    ###
    # Loop through the pixel
    for i in range(N):
        cosa = (bbox[0,1]-bbox[0,0])*(i+.5)/N+bbox[0,0]	
        for j in range(N):	
            cosb = (bbox[1,1]-bbox[1,0])*(j+.5)/N+bbox[1,0]
            
            ###
            # loop over the baselines
            pixelValue = 0
            for k in range( xc.shape[0] ):
            # for k in prange( xc.shape[0], nogil=True ):
                #the time delay for this location on this baseline (in us)
                #the delay is in ns, everything else is in us
                tau = (A[k,0]*cosa+A[k,1]*cosb)*bl[k]/C - dl[k]/1000
                #convert to the time delay in samples
                tau = tau*fs

                #refernce from the 0 time of the array
                tau = mMiddle-tau
                
                #convert to array numbered, there's a cython option 
                #which would make this not needed
                if tau < 0:
                    tau = xcM+tau

                #did we step out of the array on accident?
                if tau < 0 or tau >= xcM:
                    continue
                
                #add this to the image (and normalize)
                pixelValue +=  1#quadint( xc,k, tau )
            Output[i,j] = pixelValue/xcM
    
    return Output
