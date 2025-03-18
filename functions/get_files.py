import datetime
import xarray as xr
import numpy as np
from datetime import timedelta
import cf_xarray as cfxr
import glob
import os

def test():
    return print("Functions from get_files.py imported!")

def open_by_name(name, vars, catalog):
    """Return a dataset for the requested name and vars"""
    return (
        catalog[name]
        .search(variable=vars)
        .to_dask(
            xarray_open_kwargs={
                "chunks": {"time": "auto", "ni": -1, "nj": -1},
                "decode_coords": False,
            },
            xarray_combine_by_coords_kwargs={
                "compat": "override",
                "data_vars": "minimal",
                "coords": "minimal",
            },
        )
    )



def open_by_experiment(exp_name, vars, VARS_2D, RUN_DICT, catalog):
    """Concatenate any datasets provided for this experiment into one ds, and add area and geo coordinates"""
    if RUN_DICT[exp_name]["model"] in ["access-om2", "access-om3"]:
        # If the experiment is in the Catalog then get the data from there
        # get the data for each run of this config
        cice_ds = xr.concat(
            [open_by_name(iName, vars, catalog) for iName in [RUN_DICT[exp_name]['expt']]], dim="time"
        )
    
        # We also want the area/lat/lon fields, but these are not time dependent.
        area_ds = xr.merge(
            [
                xr.open_dataset(
                    catalog[RUN_DICT[exp_name]['expt']]
                    .search(variable=iVar)
                    .df.path[0]
                    # path of the first file with the area field, the geolon field and the geolat field
                ).drop_vars("time")
                for iVar in VARS_2D
            ]
        )
    
        # Label the lats and lons
        cice_ds.coords["ni"] = area_ds["xt_ocean"].values
        cice_ds.coords["nj"] = area_ds["yt_ocean"].values
    
        # Copy attributes for cf compliance
        cice_ds.ni.attrs = area_ds.xt_ocean.attrs
        cice_ds.nj.attrs = area_ds.yt_ocean.attrs
    
        cice_ds = cice_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean"}))
    
        # Add the geolon, geolat, and area as extra co-ordinates fields from area_t
    
        cice_ds = cice_ds.assign_coords(
            {
                "geolat_t": area_ds.geolat_t,
                "geolon_t": area_ds.geolon_t,
                "area_t": area_ds.area_t,
            }
        )
    
        # cice timestamps are also misleading:
        cice_ds["time"] = cice_ds.time.to_pandas() - timedelta(minutes=1)

    
    elif RUN_DICT[exp_name]["model"] == "access-om2-dev":
        # Else, it is a development expt and stored on ps29
        base_dir = f"/scratch/ps29/nd0349/{exp_name[0:10].lower()}/archive/{RUN_DICT[exp_name]['expt']}/"
        output_dirs = glob.glob(os.path.join(base_dir, "output[0-9][0-9][0-9]"))
        
        # Extract most recent simulation
        if output_dirs:
            output_number = max(int(d[-3:]) for d in output_dirs)  # Extract last 3 digits
            output_number = f"{output_number:03d}"  # Ensure it stays in 000 format
        else:
            output_number = "000"  # Default if no directories exist
        
        path = os.path.join(base_dir, f"output{output_number}", "ice/OUTPUT/")
        file_pattern = os.path.join(path, "iceh.????-??.nc")
        files = sorted(glob.glob(file_pattern))
        file_list = sorted(glob.glob(file_pattern))
        print(files)
        sample_ds = xr.open_mfdataset(files[0],combine="by_coords")

        # List of variables to keep
        keep_vars = ["TLAT", "TLON", "time"] + vars
        
        drop_vars = [var for var in sample_ds.variables if var not in keep_vars]
        
        cice_ds = xr.open_mfdataset(
            files, 
            combine="by_coords", 
            decode_timedelta=False,
            drop_variables=drop_vars  # Now using the correct list
        )
        
        # We also want the area/lat/lon fields, but these are not time dependent.
        area_ds = xr.open_mfdataset("/g/data/vk83/configurations/inputs/access-om3/cice/grids/global.1deg/2024.05.14/grid.nc", 
                                    combine="by_coords")
        area_ds = area_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean", "tarea": "area_t", "tlat": "geolat_t", "tlon": "geolon_t"}))
        # VARS_2D = ["area_t", "geolat_t", "geolon_t"]
        # VARS_2D = ["area_t", "geolat_t", "geolon_t"]
        
        # Label the lats and lons
        cice_ds.coords["ni"] = area_ds["xt_ocean"].values
        cice_ds.coords["nj"] = area_ds["yt_ocean"].values
        
        # Copy attributes for cf compliance
        cice_ds.ni.attrs = area_ds.xt_ocean.attrs
        cice_ds.nj.attrs = area_ds.yt_ocean.attrs
        
        cice_ds = cice_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean"}))
        
        # Add the geolon, geolat, and area as extra co-ordinates fields from area_t
        
        cice_ds = cice_ds.assign_coords(
            {
                "geolat_t": area_ds.geolat_t,
                "geolon_t": area_ds.geolon_t,
                "area_t": area_ds.area_t,
            }
        )
        
        # cice timestamps are also misleading:
        cice_ds["time"] = cice_ds.time.to_pandas() - timedelta(minutes=1)


    
    elif RUN_DICT[exp_name]["model"] == "access-om3-dev":
        # Else, it is a development expt and stored on ps29
        most_recent_file = False
        run_status = 'archive' # 'archive' or 'work'
        if run_status == 'work':
            print("Using work directory")
            base_dir = f"/scratch/ps29/nd0349/{exp_name[0:10].lower()}/work/{RUN_DICT[exp_name]['expt']}/"
            file_pattern = os.path.join(base_dir, "access-om3.cice.1mon.mean.????-??.nc")
        else:
        
            base_dir = f"/scratch/ps29/nd0349/{exp_name[0:10].lower()}/archive/{RUN_DICT[exp_name]['expt']}/"
            # base_dir = f" /g/data/ps29/nd0349/access-om2/archive/{exp_name[0:10].lower()}/archive/{RUN_DICT[exp_name]['expt']}/"
           
            output_dirs = glob.glob(os.path.join(base_dir, "output[0-9][0-9][0-9]"))
        
            # Extract most recent simulation
            if output_dirs:
                output_number = max(int(d[-3:]) for d in output_dirs)  # Extract last 3 digits
                output_number = f"{output_number:03d}"  # Ensure it stays in 000 format
            else:
                output_number = "000"  # Default if no directories exist
            
            file_pattern = os.path.join(base_dir, f"output{output_number}", "access-om3.cice.1mon.mean.????-??.nc")

        if most_recent_file:
            print(f"Output folder: {output_number}")
            files = sorted(glob.glob(file_pattern))
        else:
            all_files = []
            for output_dir in output_dirs:
                file_pattern = os.path.join(output_dir, "access-om3.cice.1mon.mean.????-??.nc")
                all_files.extend(glob.glob(file_pattern))
            files = sorted(all_files)
        
        
        sample_ds = xr.open_mfdataset(files[0],combine="by_coords")

        # List of variables to keep
        keep_vars = ["TLAT", "TLON", "time"] + vars
        
        drop_vars = [var for var in sample_ds.variables if var not in keep_vars]
        
        cice_ds = xr.open_mfdataset(
            files, 
            chunks="auto",
            combine="by_coords", 
            decode_timedelta=False,
            drop_variables=drop_vars  # Now using the correct list
        )
        
        # We also want the area/lat/lon fields, but these are not time dependent.
        area_ds = xr.open_mfdataset("/g/data/vk83/configurations/inputs/access-om3/cice/grids/global.1deg/2024.05.14/grid.nc",
                                    drop_variables=["ulat","ulon"],
                                    combine="by_coords")
        area_ds = area_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean", "tarea": "area_t", "tlat": "geolat_t", "tlon": "geolon_t"}))
        # VARS_2D = ["area_t", "geolat_t", "geolon_t"]
        # VARS_2D = ["area_t", "geolat_t", "geolon_t"]

        def radians_to_degrees(da):
            return da * (180 / np.pi)
        
        # Apply conversion efficiently while handling Dask arrays
        area_ds = area_ds.assign_coords(
            geolat_t=xr.apply_ufunc(radians_to_degrees, area_ds.geolat_t, dask="parallelized"),
            geolon_t=xr.apply_ufunc(radians_to_degrees, area_ds.geolon_t, dask="parallelized")
        )

        
        # Label the lats and lons
        cice_ds.coords["ni"] = area_ds["xt_ocean"].values
        cice_ds.coords["nj"] = area_ds["yt_ocean"].values
        
# plt.pcolormesh(all_data[1].geolon_t, all_data[1].geolat_t, all_data[1], shading='auto')

        
        # Copy attributes for cf compliance
        cice_ds.ni.attrs = area_ds.xt_ocean.attrs
        cice_ds.nj.attrs = area_ds.yt_ocean.attrs
        
        cice_ds = cice_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean"}))
        
        # Add the geolon, geolat, and area as extra co-ordinates fields from area_t
        
        cice_ds = cice_ds.assign_coords(
            {
                "geolat_t": area_ds.geolat_t,
                "geolon_t": area_ds.geolon_t,
                "area_t": area_ds.area_t,
            }
        )


        
        # cice timestamps are also misleading:
        # cice_ds["time"] = cice_ds.time.to_pandas() - timedelta(minutes=1)

    elif RUN_DICT[exp_name]["model"] == "access-om3-wav-dev":
        # ACCESS-OM3 development
        base_dir = f"/scratch/ps29/nd0349/{exp_name[0:10].lower()}/archive/{RUN_DICT[exp_name]['expt']}/"
        output_dirs = glob.glob(os.path.join(base_dir, "output[0-9][0-9][0-9]"))
        
        # # Extract most recent simulation
        # if output_dirs:
        #     output_number = max(int(d[-3:]) for d in output_dirs)  # Extract last 3 digits
        #     output_number = f"{output_number:03d}"  # Ensure it stays in 000 format
        # else:
        #     output_number = "000"  # Default if no directories exist
        
        # path = os.path.join(base_dir, f"output{output_number}")
        # file_pattern = os.path.join(path, "access-om3.cice.*.nc")
        # files = sorted(glob.glob(file_pattern))
        # file_list = sorted(glob.glob(file_pattern))

        file_pattern = os.path.join(base_dir, "output[0-9][0-9][0-9]", "access-om3.cice.*.nc")
        file_list = sorted(glob.glob(file_pattern))
        print(file_list)
        # Read in data
        if file_list:
            cice_ds = xr.open_mfdataset(file_list, combine="by_coords")
        else:
            print("No matching files found.")


        # Read in data
        # cice_ds = xr.open_mfdataset(file_list, combine="by_coords")
        # cice_ds['aice_m'] = cice_ds['aice'].resample(time='ME').max()
        cice_ds['aice_m'] = cice_ds['aice']
        area_ds = xr.open_mfdataset("/g/data/vk83/configurations/inputs/access-om3/cice/grids/global.1deg/2024.05.14/grid.nc", 
                                    combine="by_coords")
        area_ds = area_ds.rename(({"ni": "xt_ocean", 
                                   "nj": "yt_ocean", 
                                   "tarea": "area_t", 
                                   "tlat": "geolat_t", 
                                   "tlon": "geolon_t"})
                                )
        # VARS_2D = ["area_t", "geolat_t", "geolon_t"]
        
        # Label the lats and lons
        cice_ds.coords["ni"] = area_ds["xt_ocean"].values
        cice_ds.coords["nj"] = area_ds["yt_ocean"].values
        
        # Copy attributes for cf compliance
        cice_ds.ni.attrs = area_ds.xt_ocean.attrs
        cice_ds.nj.attrs = area_ds.yt_ocean.attrs
        
        cice_ds = cice_ds.rename(({"ni": "xt_ocean", "nj": "yt_ocean"}))
        
        # Add the geolon, geolat, and area as extra co-ordinates fields from area_t
        
        cice_ds = cice_ds.assign_coords(
            {
                "geolat_t": area_ds.geolat_t,
                "geolon_t": area_ds.geolon_t,
                "area_t": area_ds.area_t,
            }
        )
        
        # cice timestamps are also misleading:
        cice_ds["time"] = cice_ds.time.to_pandas() - timedelta(minutes=1)
    return cice_ds