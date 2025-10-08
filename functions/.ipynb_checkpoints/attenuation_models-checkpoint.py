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
    α = 5.35e-6 if Hs_init >= 3 else 1.5e-6
    return α, Hs_out

# -----------------------------------------------------------------------------
def MBK_2014_attenuation(periods=None):
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