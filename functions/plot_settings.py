# -----------------------------------------------------------------------------
# ACCESS-OM3 Basic Plot Settings
# Author: Noah Day (University of Melbourne), September 2025
# 
# Provides standard colour maps, projection setups, and helper functions for 
# plotting ACCESS-OM3 sea-ice, ocean, and wave fields using matplotlib and Cartopy.
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cft
import cmocean.cm as cmo
import calendar
import re
import numpy as np

# -----------------------------------------------------------------------------
plot_settings = {
    # CICE
    "aice": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "fsdrad": {"cmap": cmo.thermal, "vmin": 0, "vmax": 850},
    "hi": {"cmap": cmo.deep, "vmin": 0, "vmax": 5},
    "hs": {"cmap": cmo.amp, "vmin": 0, "vmax": 0.5},
    "iage": {"cmap": cmo.dense, "vmin": 0, "vmax": 2},
    "icespd": {"cmap": cmo.amp, "vmin": 0, "vmax": 0.5},
    "hfs": {"cmap": cmo.amp, "vmin": 0, "vmax": 1},
    "tice": {"cmap": cmo.balance, "vmin": -30, "vmax": 0},
    "uvel": {"cmap": cmo.balance, "vmin": -0.5, "vmax": 0.5},
    "vvel": {"cmap": cmo.balance, "vmin": -0.5, "vmax": 0.5},
    "strength": {"cmap": cmo.matter, "vmin": 0, "vmax": 50e3},
    "divu": {"cmap": cmo.curl, "vmin": -1e-5, "vmax": 1e-5},
    "shear": {"cmap": cmo.curl, "vmin": 0, "vmax": 1e-5},
    "Tair": {"cmap": cmo.thermal, "vmin": -30, "vmax": 40},
    "uatm": {"cmap": cmo.balance, "vmin": -20, "vmax": 20},
    "vatm": {"cmap": cmo.balance, "vmin": -20, "vmax": 20},
    "Qref": {"cmap": cmo.dense, "vmin": 0, "vmax": 25},
    "fswdn": {"cmap": cmo.solar, "vmin": 0, "vmax": 1200},
    "flwdn": {"cmap": cmo.thermal, "vmin": 100, "vmax": 500},
    "snow_ai": {"cmap": cmo.gray, "vmin": 0, "vmax": 1.5e-6},
    "snow": {"cmap": cmo.gray, "vmin": 0, "vmax": 0.5},
    "sst": {"cmap": cmo.thermal, "vmin": -2, "vmax": 30},
    "sss": {"cmap": cmo.haline, "vmin": 32, "vmax": 37},
    "uocn": {"cmap": cmo.balance, "vmin": -0.5, "vmax": 0.5},
    "vocn": {"cmap": cmo.balance, "vmin": -0.5, "vmax": 0.5},
    "wave_sig_ht": {"cmap": cmo.tempo, "vmin": 0, "vmax": 10}, 
    # MOM6
    "tos": {"cmap": cmo.thermal, "vmin": -8, "vmax": 20},
    "sos": {"cmap": cmo.haline, "vmin": 33, "vmax": 35},
    "speed": {"cmap": "viridis", "vmin": 0, "vmax": 0.5}, 
    # WW3
    "HS": {"cmap": cmo.tempo, "vmin": 0, "vmax": 10}, 
    "ICE": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "ICEH": {"cmap": cmo.deep, "vmin": 0, "vmax": 5},
    "ICEF": {"cmap": cmo.thermal, "vmin": 0, "vmax": 1700},
    "T02": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "T0M1": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "T0M1": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "FP": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "DIR": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "EF": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "IC1": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "IC5": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "THM": {"cmap": plt.cm.twilight_shifted, "vmin": -np.pi, "vmax": np.pi},
    "FP0": {"cmap": "viridis", "vmin": 0, "vmax": 0.5},

}

["Tair", "uatm", "vatm", "Qref", "fswdn", "flwdn", "snow"]

# -----------------------------------------------------------------------------
def get_plot_settings(var_name):
    # Remove common suffixes like "_m" (monthly), "_d" (daily), etc.
    base_var = re.sub(r"_(m|d|h|s|min|max|mean|std)$", "", var_name)
    
    # Check if the cleaned variable exists in settings
    return plot_settings.get(base_var, {"cmap": cmo.gray, "vmin": 0, "vmax": 1})

# -----------------------------------------------------------------------------
def basic_axis(dims, hemisphere="south", projection=ccrs.SouthPolarStereo(true_scale_latitude=-70), shape="circle"):
    if hemisphere=="south":
        projection = ccrs.SouthPolarStereo(true_scale_latitude=-70)
        extent = [-180, 180, -90, -45]
    elif hemisphere == "north":
        # projection = ccrs.NorthPolarStereo(true_scale_latitude=90)
        extent = [-180, 180, 60, 90]
        projection = ccrs.Stereographic(
            central_latitude=90.0,
            central_longitude=-45.0,
            true_scale_latitude=60.0,
            globe=ccrs.Globe(semimajor_axis=6378273.0, semiminor_axis=6356889.448910593)
            )
    elif hemisphere == "regional":
        proj = projection
        extent = [-180, 180, -90, 90]
        
    # Determine number of rows and columns for subplots
    ncols = dims[1]
    nrows = dims[0]
    number_panels = ncols*nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        subplot_kw={'projection': projection},
        figsize=(4 * ncols, 4 * nrows),
        gridspec_kw={'wspace': 0.15, 'hspace': 0.15}
    )
    
    if number_panels == 1:
        axes = np.array([[axes]])  # Ensure axes is always a 2D array
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape((nrows, ncols))
    
    for ax in axes.flat[:number_panels]:  # Only iterate over required axes
        ax.set_global()
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cft.LAND)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        if shape == "circle":
            # Make a circle plot
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
    
    # Hide unused subplots if necessary
    for ax in axes.flat[number_panels:]:
        ax.set_visible(False)
    axes = axes.flatten()
    return fig, axes, projection
    
# -----------------------------------------------------------------------------
def add_ice_contours(ax, ds_plot, hemisphere, projection):
    if ds_plot.attrs['intake_esm_attrs:realm'] == 'wave':
        lat = 'lat'
        lon = 'lon'
        ice = 'ICE'
    else:
        lat = 'TLAT'
        lon = 'TLON'
        ice = 'aice'
        
    transformed_coor=projection.transform_points(ccrs.PlateCarree(),ds_plot[lon].values,ds_plot[lat].values)
    x_ster,y_ster=transformed_coor[:,:,0],transformed_coor[:,:,1]
    use_transformed_coordinates_directly = True
    
    if use_transformed_coordinates_directly:
        cs = ax.contour(x_ster,y_ster,ds_plot[ice],levels=[1e-12, 0.15, 0.8],linestyles=[":", "-", "--"],colors="magenta",linewidths=0.5)
    else:
        cs = ax.contour(ds_plot[lon],ds_plot[lat],ds_plot[ice],levels=[1e-12, 0.15, 0.8],transform=ccrs.PlateCarree(),
                    linestyles=[":", "-", "--"],colors="magenta",linewidths=0.5)
                
    return cs

    
# -----------------------------------------------------------------------------
def add_swh_contours(ax, ds_plot, hemisphere, projection):
    if ds_plot.attrs['intake_esm_attrs:realm'] == 'wave':
        lat = 'lat'
        lon = 'lon'
        swh = 'HS'
    else:
        lat = 'TLAT'
        lon = 'TLON'
        swh = 'wave_sig_ht'

    transformed_coor=projection.transform_points(ccrs.PlateCarree(),ds_plot[lon].values,ds_plot[lat].values)
    x_ster,y_ster=transformed_coor[:,:,0],transformed_coor[:,:,1]
    use_transformed_coordinates_directly = True
    
    if use_transformed_coordinates_directly:
        cs = ax.contour(x_ster,y_ster,ds_plot[swh],levels=[1e-8, 1e-1, 1e0],linestyles=[":", "-", "--"],colors="black",linewidths=0.5)
    else:
        cs = ax.contour(ds_plot[lon],ds_plot[lat],ds_plot[swh],levels=[1e-8, 1e-1, 1e0],transform=ccrs.PlateCarree(),
                    linestyles=[":", "-", "--"],colors="black",linewidths=0.5)
                
    return cs