# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cft
import cmocean.cm as cmo
import matplotlib.lines as mlines
import cartopy.feature as cft
import matplotlib.colors as mcolors
import calendar
import re

plot_settings = {
    "aice": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
    "ICE": {"cmap": cmo.ice, "vmin": 0, "vmax": 1},
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
    "HS": {"cmap": cmo.tempo, "vmin": 0, "vmax": 10}, 
}

["Tair", "uatm", "vatm", "Qref", "fswdn", "flwdn", "snow"]

# Function to get plot settings for any variable
def get_plot_settings(var_name):
    # Remove common suffixes like "_m" (monthly), "_d" (daily), etc.
    base_var = re.sub(r"_(m|d|h|s|min|max|mean|std)$", "", var_name)
    
    # Check if the cleaned variable exists in settings
    return plot_settings.get(base_var, {"cmap": cmo.gray, "vmin": 0, "vmax": 1})


def basic_axis(dims, hemisphere="south"):
    if hemisphere=="south":
        projection = ccrs.SouthPolarStereo(true_scale_latitude=-70)
        extent = [-180, 180, -90, -45]
    else:
        # projection = ccrs.NorthPolarStereo(true_scale_latitude=90)
        extent = [-180, 180, 60, 90]
        projection = ccrs.Stereographic(
            central_latitude=90.0,
            central_longitude=-45.0,
            true_scale_latitude=60.0,
            globe=ccrs.Globe(semimajor_axis=6378273.0, semiminor_axis=6356889.448910593)
            )
        
    # Determine number of rows and columns for subplots
    ncols = dims[1]#int(np.ceil(np.sqrt(number_panels)))
    nrows = dims[0]#int(np.ceil(number_panels / ncols))
    number_panels = ncols*nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        subplot_kw={'projection': projection},
        figsize=(4 * ncols, 4 * nrows),  # Adjust figure size
        gridspec_kw={'wspace': 0.15, 'hspace': 0.15}  # Adjust spacing
    )
    
    if number_panels == 1:
        axes = np.array([[axes]])  # Ensure axes is always a 2D array
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape((nrows, ncols))
    
    land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='gray', linewidth=0.5)
    
    for ax in axes.flat[:number_panels]:  # Only iterate over required axes
        ax.set_global()
        ax.coastlines(resolution='50m')
        ax.add_feature(land_50m)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Hide unused subplots if necessary
    for ax in axes.flat[number_panels:]:
        ax.set_visible(False)
    axes = axes.flatten()
    return fig, axes, projection

def add_ice_contours(ax, ds_plot, hemisphere, projection):
    # if hemisphere=="north":
    transformed_coor=projection.transform_points(ccrs.PlateCarree(),ds_plot['TLON'].values,ds_plot['TLAT'].values)
    x_ster,y_ster=transformed_coor[:,:,0],transformed_coor[:,:,1]
    use_transformed_coordinates_directly = True
    # inner_Arctic=xr.where(ds_plot['TLAT']>=81,1,0)
    
    if use_transformed_coordinates_directly:
        cs = ax.contour(x_ster,y_ster,ds_plot['aice'],levels=[0.15, 0.8],linestyles=["-", "--"],colors="magenta",linewidths=0.5)
    else:
        cs = ax.contour(ds_plot['TLON'],ds_plot['TLAT'],ds_plot['aice'],levels=[0.15, 0.8],transform=ccrs.PlateCarree(),
                    linestyles=["-", "--"],colors="magenta",linewidths=0.5)
                
    return cs
