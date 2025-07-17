import xarray as xr
import numpy as np
from tqdm import tqdm

def ciceBinWidths(NFSD):
    """
    Return the bin widths used in CICE.

    Args:
        NFSD (ndarray): Floe bin centres [m]

    Returns:
        BinWidths (ndarray): Binwidths [m]
        BinLeft (ndarray): Binwidths [m]
        BinRight (ndarray): Binwidths [m]
    """
    min_floe = 6.65000000e-02;
    # Define arrays
    fsd_lims = np.zeros(len(NFSD)+1)
    floe_rad_c = np.zeros(len(NFSD))
    floe_rad_l = np.zeros(len(NFSD))
    floe_rad_h = np.zeros(len(NFSD))
    
    fsd_lims[1] = min_floe;
    for i in range(len(NFSD)):
        fsd_lims[i+1] = 2*NFSD[i] - fsd_lims[i]
        floe_rad_c[i] = NFSD[i]
    
    
    floe_rad_l = fsd_lims[0:-1]
    floe_rad_h = fsd_lims[1:]
    floe_binwidth = floe_rad_h - floe_rad_l
    
    floeshape = 0.66;
    floe_area_l = 4*floeshape*floe_rad_l**2
    floe_area_c = 4*floeshape*floe_rad_c**2
    floe_area_h = 4*floeshape*floe_rad_h**2
    
    BinWidths = floe_rad_h - floe_rad_l
    BinRight = floe_rad_h
    BinLeft = floe_rad_l

    return BinWidths, BinLeft, BinRight


# def integrateFloeSize(ds, var):
#     ds = ds.where(ds['aice'] > 0.01, drop=False)
#     var = 'dafsd_wave'
#     data3d = ds[var][:,:,:].values
#     data = np.zeros(ds.TLON.shape)
#     NFSD = ds.NFSD.values
#     floe_binwidth, _, _ = ciceBinWidths(NFSD)

#     for nf in range(0,len(floe_binwidth)):
#         data += data3d[nf,:,:]*NFSD[nf]/ds.aice[:,:].values
#     return data

def integrateFloeSize(ds, var):
    tmp_var = var[0:-3]
    puny = 10**-12
    int_var = ds[tmp_var].isel(nf=1)
    int_var = int_var.where(ds['aice'] > puny, drop=False)
    NFSD = ds.NFSD.values
    floe_binwidth, _, _ = ciceBinWidths(NFSD)
    ds_out = ds.copy()
    # Put in temporary data
    ds_out[var] = ds_out[tmp_var].isel(nf=0)
    ds_out[var][:] = 0.0
    int_values = np.zeros(int_var.shape)
           
    for idx, time in tqdm(enumerate(ds.time), total = len(ds.time.values), desc='Integrating over floe size space'):
        tmp_data = np.zeros(ds.TLON.shape)
        ds_tmp = ds.where(ds['aice'] > puny, drop=False).sel(time=time)
        for nf in range(0,len(floe_binwidth)):
            tmp_data += ds_tmp[tmp_var].isel(nf=nf).values*NFSD[nf]/ds_tmp.aice[:,:].values
        int_var[idx,:,:] = tmp_data
        # print(np.nanmin(tmp_data))
        # print(np.nanmin(int_var[idx,:,:]))
    ds_out[var].values = int_var
    # print(np.nanmin(ds_out[var].isel(time=0).values))
    
    # Set attributes
    parts = ds[tmp_var].attrs['long_name'].split(':')
    part1 = parts[0]
    part2 = parts[1]
    ds_out[var].attrs['long_name'] = part1[:-3] + 'rep. radius:' + part2
    ds_out['dafsd_wave_ra'].attrs['units'] = 'm/timestep'

    return ds_out