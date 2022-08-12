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

def test( float[:] arr ):

    cdef int N = <int> arr.shape[0]
    cdef np.ndarray[float, ndim=2] Output = np.zeros( (N,N), dtype='float32' )
    cdef float x
    cdef int i,j
    for i in prange(N, nogil=True):
        x = arr[i]
        for j in range(N):
            Output[i,j] = x + arr[j] 
    
    return Output


# ###
# # for correlating
# cdef fpad( complex[:] X, int M):
#     """fpad(X,M)
#     Frequency 0 pads X to be M*len(X) long,
#     Used for fft based interpolation
#     input - 
#         X	-	fft of a signal, X(f)
#         M	-	factor to interpolate by
#     output -
#         padded output(f) 
#     """

#     if M  <= 1:
#         return X
#     N = len(X)
     
#     ###
#     # create the output array
#     output = np.zeros( N*M , dtype='complex')
#     #the first N/2 samples
#     output[:N//2] = X[:N//2]*M
#     output[N//2]  = X[N//2]/2*M
#     #the last N/2 samples
#     output[-N//2] = X[N//2]/2*M
#     output[-N//2+1:] = X[N//2+1:]*M
     
#     return output

###
# For imaging
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

def pimage( float[:,:] xc, 
            float[:] bl, 
            float[:] dl,
            float[:,:] A, 
            int N=50, 
            float fs=200,
            float[:,:] bbox = np.array( [[-1.1,1.1],[-1.1,1.1]], dtype='float32' ),
            float C=299.792458
            ):

    """
    A parallel capable version of image.  
    Inputs and outputs are identical to the non-parallel version, this one just melts CPUs faster
    Performance gains depend on how many CPUs your system has, and if they're already in use
    Parallel performance now actually better than single threaded performance!

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

    cdef float cosa, cosb, tau, dtau, az, el

    #these are used to calculate the amplitude of the pixel
    cdef float p, l, a
        
    #counters, there's a bunch of them
    cdef int i,j,k,m

    cdef int j0,j1,j2   #for quadint

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
    for i in prange(N, nogil=True):
        cosa = (bbox[0,1]-bbox[0,0])*(i+.5)/N+bbox[0,0]	
        for j in range(N):
            cosb = (bbox[1,1]-bbox[1,0])*(j+.5)/N+bbox[1,0]

            ###
            # loop over the baselines
            pixelValue = 0
            # for k in prange( xcN, nogil=True ):
            for k in range( xcN ):
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

                ###
                # This is the quadint portion of the code.
                # it's been pulled out here to make the parellel stuff easier to work with 
                j0 = <int> tau
                #these are to index the array
                #we generate them here so that the can wrap 
                #properly if needed
                j1 = j0+1 
                j2 = j0+2
                if j1 >= xcM:
                    j1 = j1-xcM
                if j2 >= xcM:
                    j2 = j2-xcM
                
                # cdef float Output = 0, a
                # #there are 3 terms
                a = .5*(tau-j0-1)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j0]/xcN
                a = -1*(tau-j0)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j1]/xcN
                a = .5*(tau-j0)*(tau-j0-1)
                pixelValue = pixelValue + a*xc[k,j2]/xcN
                

                #add this to the image (and normalize)
                # pixelValue +=  quadint( xc,k, tau )
            Output[i,j] = pixelValue
    
    return Output

def wimage( float[:,:] xc, 
            float[:] bl, 
            float[:] dl,
            float[:,:] A, 
            float[:] wsp,
            int N=50, 
            float fs=200,
            float[:,:] bbox = np.array( [[-1.1,1.1],[-1.1,1.1]], dtype='float32' ),
            float C=299.792458
            ):

    """
    A parallel capable version of image.  
    Inputs and outputs are identical to the non-parallel version, this one just melts CPUs faster
    Performance gains depend on how many CPUs your system has, and if they're already in use
    Parallel performance now actually better than single threaded performance!

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

    cdef float cosa, cosb, tau, dtau, az, el

    #these are used to calculate the amplitude of the pixel
    cdef float p, l, a, W
        
    #counters, there's a bunch of them
    cdef int i,j,k,m

    cdef int j0,j1,j2   #for quadint

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
    for i in prange(N, nogil=True):
        cosa = (bbox[0,1]-bbox[0,0])*(i+.5)/N+bbox[0,0]	
        for j in range(N):
            cosb = (bbox[1,1]-bbox[1,0])*(j+.5)/N+bbox[1,0]

            ###
            # loop over the baselines
            pixelValue = 0
            W = 0
            # for k in prange( xcN, nogil=True ):
            for k in range( xcN ):
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

                ###
                # This is the quadint portion of the code.
                # it's been pulled out here to make the parellel stuff easier to work with 
                j0 = <int> tau
                #these are to index the array
                #we generate them here so that the can wrap 
                #properly if needed
                j1 = j0+1 
                j2 = j0+2
                if j1 >= xcM:
                    j1 = j1-xcM
                if j2 >= xcM:
                    j2 = j2-xcM
                
                # cdef float Output = 0, a
                # #there are 3 terms
                a = .5*(tau-j0-1)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j0]*wsp[k]
                a = -1*(tau-j0)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j1]*wsp[k]
                a = .5*(tau-j0)*(tau-j0-1)
                pixelValue = pixelValue + a*xc[k,j2]*wsp[k]

                W += wsp[k]
                

                #add this to the image (and normalize)
                # pixelValue +=  quadint( xc,k, tau )
            Output[i,j] = pixelValue/wsp[k]
    
    return Output



def pimage_azel( float[:,:] xc, 
            float[:] bl, 
            float[:] dl,
            float[:,:] A, 
            int N=50, 
            float fs=200,
            float[:,:] bbox = np.array( [[-180,180],[0,90]], dtype='float32' ),
            float C=299.792458
            ):

    """
    A parallel capable version of image_azel (which doesn't exist).  
    Image is computed in azimuth/elevation bounds, instead of in the cosine projection
    Inputs and outputs are identical to the non-parallel version, this one just melts CPUs faster
    Performance gains depend on how many CPUs your system has, and if they're already in use

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

    cdef float cosa, cosb, tau, dtau, az, el

    #these are used to calculate the amplitude of the pixel
    cdef float p, l, a
        
    #counters, there's a bunch of them
    cdef int i,j,k,m

    cdef int j0,j1,j2   #for quadint

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
    for i in prange(N, nogil=True):
        az = (bbox[0,1]-bbox[0,0])*(i+.5)/N+bbox[0,0]	
        for j in range(N):
            el = (bbox[1,1]-bbox[1,0])*(j+.5)/N+bbox[1,0]

            #convert az/el to cosa,cosb
            cosa = sin( az * M_PI/180. ) * cos( el *M_PI/180. )
            cosb = cos( az * M_PI/180. ) * cos( el *M_PI/180. )
            
            ###
            # loop over the baselines
            pixelValue = 0
            # for k in prange( xcN, nogil=True ):
            for k in range( xcN ):
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

                ###
                # This is the quadint portion of the code.
                # it's been pulled out here to make the parellel stuff easier to work with 
                j0 = <int> tau
                #these are to index the array
                #we generate them here so that the can wrap 
                #properly if needed
                j1 = j0+1 
                j2 = j0+2
                if j1 >= xcM:
                    j1 = j1-xcM
                if j2 >= xcM:
                    j2 = j2-xcM
                
                # cdef float Output = 0, a
                # #there are 3 terms
                a = .5*(tau-j0-1)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j0]/xcN
                a = -1*(tau-j0)*(tau-j0-2)
                pixelValue = pixelValue + a*xc[k,j1]/xcN
                a = .5*(tau-j0)*(tau-j0-1)
                pixelValue = pixelValue + a*xc[k,j2]/xcN
                

                #add this to the image (and normalize)
                # pixelValue +=  quadint( xc,k, tau )
            Output[i,j] = pixelValue
    
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
            for k in range( xcN ):
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
                pixelValue +=  quadint( xc,k, tau )/xcN
            Output[i,j] = pixelValue
    
    return Output
