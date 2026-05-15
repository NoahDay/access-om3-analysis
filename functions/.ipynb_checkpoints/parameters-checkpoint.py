import numpy as np

𝛑 = np.pi
# √ = np.sqrt()

# --- Ice strength parameters ---
SIGMA_0_MPA = 1.76
SIGMA_0 = SIGMA_0_MPA * 1e6  # Pa
# SIGMA_C = SIGMA_0 * np.exp(-5.88 * np.sqrt(V_B))  # Williams et al. (2012)

V_B = 0.1            # Brine volume fraction (Williams et al., 2013b)
SIGMA_C = 0.27e9     # Flexural strength [Pa] (Williams et al., 2013b)
EPS_C = 4.99e-5      # Breaking strain [-] (Williams et al., 2013b)
E_C = 7.05e-5        # Breaking signficant strain [-] (Williams et al., 2013b)

# --- Physical constants ---
RHO_W = 1025.0       # Density of seawater [kg/m^3] (Williams et al., 2013b)
RHO_I = 922.5        # Density of sea ice [kg/m^3] (Williams et al., 2013b)
GRAVITY = 9.81       # Gravitational acceleration [m/s^2] (Williams et al., 2013b)

# --- Elastic properties ---
Y_0 = 5.5e9          # Effective Young's modulus [Pa] (Williams et al., 2013b)
NU = 0.295           # Poisson's ratio [-] (Williams et al., 2013b)

# --- Viscosity ---
VISC_RP = 13.0       # Robinson Palmer viscosity damping parameter [Pa s/m]