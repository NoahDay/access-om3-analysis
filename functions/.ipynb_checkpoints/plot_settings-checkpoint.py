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
}

["Tair", "uatm", "vatm", "Qref", "fswdn", "flwdn", "snow"]

# Function to get plot settings for any variable
def get_plot_settings(var_name):
    # Remove common suffixes like "_m" (monthly), "_d" (daily), etc.
    base_var = re.sub(r"_(m|d|h|s|min|max|mean|std)$", "", var_name)
    
    # Check if the cleaned variable exists in settings
    return plot_settings.get(base_var, {"cmap": cmo.gray, "vmin": 0, "vmax": 1})
