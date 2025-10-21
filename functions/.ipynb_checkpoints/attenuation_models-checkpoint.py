# -----------------------------------------------------------------------------
# Various wave–ice attenuation models and profiles
# Author: Noah Day (University of Melbourne), October 2025
# 
# Provides standard colour maps, projection setups, and helper functions for 
# plotting ACCESS-OM3 sea-ice, ocean, and wave fields using matplotlib and Cartopy.
# -----------------------------------------------------------------------------
import numpy as np

def get_ww3_freqs(nk=25):
    """
    Generate WW3 frequency bins and corresponding periods.
    
    Parameters
    ----------
    nk : float
        Number of wave numbers/frequency bins.
    """
    f1 = 0.04118   # lowest frequency [Hz]
    xfr = 1.1      # frequency growth factor
    
    freqs = f1 * xfr**np.arange(nk)
    periods = 1.0 / freqs
    return freqs, periods

# -----------------------------------------------------------------------------
def kohout_2014_attenuation(Hs_init=1.0):
    """
    Calculate wave height attenuation through sea ice following Kohout et al. (2014).
    
    Parameters
    ----------
    Hs_init : float
        Initial significant wave height [m].
    
    Returns
    -------
    α : float
        Attenuation coefficient [1/m].
    """
    α = 5.35e-6 if Hs_init <= 3 else 16.05e-6
    return α

# -----------------------------------------------------------------------------
def mbk_2014_attenuation(periods=None):
    """
    Calculate wave attenuation coefficients following Meylan, Bennetts, and Kohout (2014)
    as a function of wave period.
    
    Parameters
    ----------
    periods : array-like, optional
        Wave periods [s]. If None, uses WW3 default frequency bins.
    
    Returns
    -------
    α : np.ndarray
        Attenuation coefficients [1/m] for each period.
    """
    if periods is None:
        _, periods = get_ww3_freqs()
    
    a = 2.2e-3
    b = 4.59e-2
    α = a / periods**2 + b / periods**4
    return α

# -----------------------------------------------------------------------------
def meylan_2021_attenuation(periods, thickness, floeDiameter):
    # Fortran code from WW3 https://github.com/ACCESS-NRI/WW3/blob/dev/2025.08/model/src/w3sic4md.F90
    # !CMB added option of cubic fit to Meylan, Horvat & Bitz in prep
    #       ! ICECOEF1 is thickness
    #       ! ICECOEF5 is floe size
    #       ! TPI/SIG is period
    #       x3=min(ICECOEF1,3.5)        ! limit thickness to 3.5 m
    #       x3=max(x3,0.1)        ! limit thickness >0.1 m since I make fit below
    #       x2=min(ICECOEF5*0.5,100.0)  ! convert dia to radius, limit to 100m
    #       x2=max(2.5,x2)
    #       x2sqr=x2*x2
    #       x3sqr=x3*x3
    #       amhb = 2.12e-3
    #       bmhb = 4.59e-2
    
    #       DO IK=1, NK
    #         x1=TPI/SIG(IK)   ! period
    #         x1sqr=x1*x1
    #         KARG1(ik)=-0.26982 + 1.5043*x3 - 0.70112*x3sqr + 0.011037*x2 +  &
    #              (-0.0073178)*x2*x3 + 0.00036604*x2*x3sqr + &
    #              (-0.00045789)*x2sqr + 1.8034e-05*x2sqr*x3 + &
    #              (-0.7246)*x1 + 0.12068*x1*x3 + &
    #              (-0.0051311)*x1*x3sqr + 0.0059241*x1*x2 + &
    #              0.00010771*x1*x2*x3 - 1.0171e-05*x1*x2sqr + &
    #              0.0035412*x1sqr - 0.0031893*x1sqr*x3 + &
    #              (-0.00010791)*x1sqr*x2 + &
    #              0.00031073*x1**3 + 1.5996e-06*x2**3 + 0.090994*x3**3
    #         KARG1(ik)=min(karg1(ik),0.0)
    #         ALPHA(ik)  = 10.0**KARG1(ik)
    #         perfour=x1sqr*x1sqr
    #         if ((x1.gt.5.0) .and. (x1.lt.20.0)) then
    #           ALPHA(IK) = ALPHA(IK) + amhb/x1sqr+bmhb/perfour
    #         else if (x1.gt.20.0) then
    #           ALPHA(IK) = amhb/x1sqr+bmhb/perfour
    #         endif
    #         WN_I(IK) = ALPHA(IK) * 0.5
    #       end do
    ICECOEF1 = thickness
    ICECOEF5 = floeDiameter
    x3 = np.minimum(ICECOEF1, 3.5)
    x3 = np.maximum(x3, 0.1)
    x2 = np.minimum(ICECOEF5*0.5, 100.0)  # convert dia to radius, limit to 100m
    x2 = np.maximum(2.5, x2)
    x2sqr = x2 * x2
    x3sqr = x3 * x3
    amhb = 2.12e-3
    bmhb = 4.59e-2

    nk = len(periods)
    KARG1 = np.zeros(nk)
    ALPHA = np.zeros(nk)
    WN_I = np.zeros(nk)
    
    for ik in range(nk):
        x1 = periods[ik]
        x1sqr = x1 * x1
        KARG1[ik] = (
            -0.26982
            + 1.5043 * x3
            - 0.70112 * x3sqr
            + 0.011037 * x2
            - 0.0073178 * x2 * x3
            + 0.00036604 * x2 * x3sqr
            - 0.00045789 * x2sqr
            + 1.8034e-05 * x2sqr * x3
            - 0.7246 * x1
            + 0.12068 * x1 * x3
            - 0.0051311 * x1 * x3sqr
            + 0.0059241 * x1 * x2
            + 0.00010771 * x1 * x2 * x3
            - 1.0171e-05 * x1 * x2sqr
            + 0.0035412 * x1sqr
            - 0.0031893 * x1sqr * x3
            - 0.00010791 * x1sqr * x2
            + 0.00031073 * x1**3
            + 1.5996e-06 * x2**3
            + 0.090994 * x3**3
        )
        
        KARG1[ik] = np.minimum(KARG1[ik], 0.0)
        ALPHA[ik] = 10.0**KARG1[ik]
        perfour = x1sqr * x1sqr
        if 5.0 < x1 < 20.0:
            ALPHA[ik] += amhb / x1sqr + bmhb / perfour
        elif x1 > 20.0:
            ALPHA[ik] = amhb / x1sqr + bmhb / perfour
        
        WN_I[ik] = ALPHA[ik] * 0.5

    α = ALPHA
    k_imaginary = WN_I
    return α, k_imaginary

