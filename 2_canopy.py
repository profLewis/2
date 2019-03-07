import numpy as np

def kmodel(x,tau):
    '''
    Filter function
    '''
    n = len(x)
    
    y = ([np.exp(-np.abs(x[i]*tau)) for i in xrange(n)]).sum()
    
    # normalise   
    y /= float(n)
    #y /= y[0]
    return y

def makeCanopy(tau,omega,xi,omegaSoil,\
               nsd=5,theta=45,dt=0.01,\
               correct=False,plot=False,diffuse=True):
    '''
    Create arrays and filters for numerical 2 stream radiative
    transfer modelling
    
    Parameters:
    
        tau       : canopy optical thickness
        omega     : canopy single scattering albedo
        xi        : canopy asymmetry factor (-1 to 1) where 0 is isotropic
                    -1 is full backscatter, +1 is full forward scatter
        omegaSoil : soil reflectance
    
    Option Values:
    
        theta : incident radiation zenith angle (degrees)
                default: 45 degrees
        dt    : step in optical thickness
                default 0.01
        nsd   : Number of std dev to extend filters
                default 5
        
    Option Flags:
    
        diffuse : downward flux is diffuse (boolean)
                  default: True
        correct : NOT IMPLEMENTED
                  whether to apply asymmetry correction term
                  default: False
        plot    : whether to produce plots
                  default: False
                  
    
    Return values:
    
        stateArrays  = (canopyI0,canopyOmega)
                       nz-sized arrays of I and omega
                       
        parameters   = (tau,omega,t_w)
                       parameters of case (possibly transformed)
                       
        domainArrays = (up,source,down)
                       nz-sized bool arrays defining domains
                       
        tauArrays    = (tauCanopy,tauFilt)
                       tau values for each element of arrays, for cxanopy
                       and filter
        filterArrays = (filterUp,filterDown)
                       Filters

        return:
        
            stateArrays,parameters,domainArrays,tauArrays,filterArrays


    
    '''

    # store setup values
    tau0       = tau
    omega0     = omega
    xi0        = xi
    omegaSoil0 = omegaSoil
    
    '''if correct:
        # corrections for xi
        tau = (1 - xi*omega0)*tau0
        omega = (1-xi)*omega0/(1 - xi * omega0)'''


    
    # how many layers in canopy: nz_canopy
    nz_canopy = int(np.ceil(tau/(dt)))
     
    # need to know filter width   : nz_filter
    # value of tau for the filter : tauFilt
    tauFilt = np.arange(-nsd,nsd,dt)
    nz_filter = len(tauFilt)

    # need to know total extent to store: nz
    nz = int(nz_canopy+2*nz_filter)
    nz = 2*int(nz/2) 
    
    # create canopy array of zeros : canopyI0
    tauCanopy = np.arange(nz)*dt - nz_filter*dt
    canopyI0 = np.zeros_like(tauCanopy)

    # source term: put unity at top of 
    # canopy: index nz_filter-1
    canopyI0[tauCanopy==0] = 1.0

    # define the canopy extent
    canopyExtent = np.logical_and(tauCanopy>0,tauCanopy<=tau)
    
    # the canopy extent has values of omega from
    # top of the canopy (index nz_filter) to
    # bottom of the canopy (index nz_filter+nz_canopy)
    canopyOmega = np.zeros_like(canopyI0)
    canopyOmega[canopyExtent] = omega
                             
    # the filter
    filterUp = np.zeros(nz_filter)
    if diffuse:
        filterUp[tauFilt<=0] = kmodel(tauFilt[tauFilt<=0],tau)
    else:
        mu = np.cos(theta*np.pi/180.)
        filterUp[tauFilt<=0] = np.exp(tauFilt[tauFilt<=0]/mu)
    
    filterDown = np.zeros_like(tauFilt)
    if diffuse:
        filterDown[tauFilt>0] = kmodel(kparams,tauFilt[tauFilt>0])
    else:
        filterDown[tauFilt>0] = np.exp(-tauFilt[tauFilt>0]/mu)
    
    # normalise the filter
    if filterUp.sum() > 0:
        filterUp /= filterUp.sum()
    if filterDown.sum() > 0:
        filterDown /= filterDown.sum()
    
    t_w = (xi/2. + 0.5)
    #t_w = xi        
    
    # define up and down domains
    up = np.zeros_like(canopyOmega).astype(bool)
    up[tauCanopy<0] = True
    down = np.zeros_like(canopyOmega).astype(bool)
    down[tauCanopy>tau] = True
    # define source domain
    source = canopyExtent
    
    stateArrays  = (canopyI0,canopyOmega)
    parameters   = (tau,omega,t_w)
    domainArrays = (up,source,down)
    tauArrays    = (tauCanopy,tauFilt)
    filterArrays = (filterUp,filterDown)
    
    return stateArrays,parameters,domainArrays,tauArrays,filterArrays
