#!/usr/bin/env python3
"""Calculate daily ERA5 Arctic/Antarctic sea ice area and extent by decade."""

import argparse
import gc
import os
import re
from glob import glob
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


EXPECTED_VARS = {
    "arctic_sie",
    "antarctic_sie",
    "arctic_sia",
    "antarctic_sia",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate daily ERA5 sea ice area and extent time series, "
            "processing and saving one decade at a time."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="/g/data/rt52/era5/single-levels/reanalysis/ci/*/ci_era5_oper_sfc_*.nc",
        help="Glob for ERA5 sea ice concentration files.",
    )
    parser.add_argument(
        "--outdir",
        default="/g/data/ps29/nd0349/datasets/ERA5",
        help="Directory for output NetCDF files.",
    )
    parser.add_argument(
        "--prefix",
        default="era5_sea_ice_area_extent_daily",
        help="Output file prefix. Files are written as PREFIX_YYYY-YYYY.nc.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year to process. Default is inferred from input files.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to process. Default is inferred from input files.",
    )
    parser.add_argument(
        "--years-per-file",
        type=int,
        default=10,
        help="Number of years per output file.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=24,
        help="Dask chunk size along time. Use 24 for roughly daily chunks.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=min(int(os.environ.get("PBS_NCPUS", 4)), 4),
        help="Number of Dask worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing complete decade files.",
    )
    return parser.parse_args()


def file_year(path):
    matches = re.findall(r"(?:19|20)\d{2}", str(path))
    if not matches:
        raise ValueError(f"No year found in path: {path}")
    return int(matches[-1])


def find_files(input_glob):
    files = [Path(p) for p in sorted(glob(input_glob))]

    if not files:
        raise FileNotFoundError(f"No ERA5 sea ice concentration files found: {input_glob}")

    files_by_year = {}
    for path in files:
        year = file_year(path)
        files_by_year.setdefault(year, []).append(path)

    print(f"Found {len(files)} ERA5 files")
    print(f"First file: {files[0]}")
    print(f"Last file:  {files[-1]}")
    print(f"Years found: {min(files_by_year)}-{max(files_by_year)}")
    return files_by_year


def year_blocks(start_year, end_year, years_per_file):
    year = start_year
    while year <= end_year:
        block_end = min(year + years_per_file - 1, end_year)
        yield year, block_end
        year = block_end + 1


def decade_outfile(outdir, prefix, start_year, end_year):
    return Path(outdir) / f"{prefix}_{start_year}-{end_year}.nc"


def existing_file_is_complete(outfile):
    if not outfile.exists():
        return False

    with xr.open_dataset(outfile) as ds_existing:
        missing = EXPECTED_VARS - set(ds_existing.data_vars)
        has_time = "time" in ds_existing.coords and ds_existing.sizes.get("time", 0) > 0
        if missing or not has_time:
            if missing:
                print(f"Existing file is incomplete. Missing variables: {sorted(missing)}")
            if not has_time:
                print("Existing file has no usable time coordinate.")
            return False

        print(f"File already exists and looks complete: {outfile}")
        print(ds_existing)
        return True


def open_era5(files, time_chunk):
    file_strings = [str(path) for path in files]

    ds_era = xr.open_mfdataset(
        file_strings,
        combine="by_coords",
        parallel=True,
        chunks={"time": time_chunk},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    return ds_era.chunk(
        {
            "time": time_chunk,
            "latitude": -1,
            "longitude": -1,
        }
    )


def get_sic(ds_era):
    if "siconc" in ds_era:
        return ds_era["siconc"]
    if "ci" in ds_era:
        return ds_era["ci"]

    raise ValueError(f"No SIC variable found. Variables are: {list(ds_era.data_vars)}")


def grid_cell_area_km2(sic, ds_era):
    radius_km = 6371.0

    lat = ds_era.latitude
    lon = ds_era.longitude

    dlat = np.abs(float(lat.diff("latitude").mean()))
    dlon = np.abs(float(lon.diff("longitude").mean()))

    lat_lower = (lat - dlat / 2).clip(min=-90, max=90)
    lat_upper = (lat + dlat / 2).clip(min=-90, max=90)

    cell_area = (
        radius_km**2
        * np.deg2rad(dlon)
        * np.abs(np.sin(np.deg2rad(lat_upper)) - np.sin(np.deg2rad(lat_lower)))
    )

    cell_area = cell_area.broadcast_like(sic.isel(time=0)).astype("float32")
    cell_area.name = "cell_area"
    cell_area.attrs["units"] = "km2"
    return cell_area


def calculate_daily_area_extent(ds_era):
    sic = get_sic(ds_era).astype("float32")
    ice = sic > 0.15

    cell_area = grid_cell_area_km2(sic, ds_era)
    area_arctic = cell_area.where(ds_era.latitude > 0, 0)
    area_antarctic = cell_area.where(ds_era.latitude < 0, 0)

    ds_hourly = xr.Dataset(
        {
            "arctic_sie": area_arctic.where(ice, 0).sum(["latitude", "longitude"]),
            "antarctic_sie": area_antarctic.where(ice, 0).sum(["latitude", "longitude"]),
            "arctic_sia": (sic * area_arctic).where(ice, 0).sum(
                ["latitude", "longitude"]
            ),
            "antarctic_sia": (sic * area_antarctic).where(ice, 0).sum(
                ["latitude", "longitude"]
            ),
        }
    )

    ds_daily = ds_hourly.resample(time="1D").mean("time")

    ds_daily["arctic_sie"].attrs = {
        "long_name": "Daily mean Arctic sea ice extent",
        "units": "km2",
        "description": "Daily mean of hourly Northern Hemisphere grid-cell area where SIC > 0.15",
    }
    ds_daily["antarctic_sie"].attrs = {
        "long_name": "Daily mean Antarctic sea ice extent",
        "units": "km2",
        "description": "Daily mean of hourly Southern Hemisphere grid-cell area where SIC > 0.15",
    }
    ds_daily["arctic_sia"].attrs = {
        "long_name": "Daily mean Arctic sea ice area",
        "units": "km2",
        "description": "Daily mean of hourly SIC-weighted Northern Hemisphere grid-cell area where SIC > 0.15",
    }
    ds_daily["antarctic_sia"].attrs = {
        "long_name": "Daily mean Antarctic sea ice area",
        "units": "km2",
        "description": "Daily mean of hourly SIC-weighted Southern Hemisphere grid-cell area where SIC > 0.15",
    }

    ds_daily.attrs["source"] = "ERA5 single-levels sea ice concentration"
    ds_daily.attrs["threshold"] = "SIC > 0.15"
    ds_daily.attrs["area_units"] = "km2"
    ds_daily.attrs["method"] = (
        "Hourly sea ice area and extent were calculated first, then averaged "
        "to daily means."
    )
    return ds_daily


def write_output(ds_out, outfile):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    tmpfile = outfile.with_suffix(".tmp.nc")

    if tmpfile.exists():
        tmpfile.unlink()

    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (365,),
        }
        for var in ds_out.data_vars
    }

    print(ds_out)

    with ProgressBar():
        ds_out.to_netcdf(tmpfile, encoding=encoding)

    tmpfile.replace(outfile)

    print(f"Saved: {outfile}")
    with xr.open_dataset(outfile) as ds_check:
        print(ds_check)
        print("Variables:", list(ds_check.data_vars))


def process_block(files_by_year, start_year, end_year, args):
    outfile = decade_outfile(args.outdir, args.prefix, start_year, end_year)

    if outfile.exists():
        if args.overwrite:
            print(f"Overwriting existing file: {outfile}")
            outfile.unlink()
        elif existing_file_is_complete(outfile):
            return
        else:
            print(f"Removing incomplete output file: {outfile}")
            outfile.unlink()

    files = []
    for year in range(start_year, end_year + 1):
        files.extend(files_by_year.get(year, []))

    if not files:
        print(f"No files found for {start_year}-{end_year}; skipping")
        return

    print(f"Processing {start_year}-{end_year} using {len(files)} files")
    ds_era = open_era5(files, args.time_chunk)
    ds_era = ds_era.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31 23:59:59"))

    ds_out = calculate_daily_area_extent(ds_era)
    write_output(ds_out, outfile)

    ds_era.close()
    ds_out.close()
    gc.collect()


def main():
    args = parse_args()

    print("Running ERA5 daily sea ice area/extent calculation")
    print(f"Output directory: {args.outdir}")
    print(f"Output prefix: {args.prefix}")
    print(f"Dask worker threads: {args.n_workers}")
    print(f"Time chunk: {args.time_chunk}")
    print(f"Years per file: {args.years_per_file}")

    dask.config.set(scheduler="threads", num_workers=args.n_workers)

    files_by_year = find_files(args.input_glob)
    start_year = args.start_year if args.start_year is not None else min(files_by_year)
    end_year = args.end_year if args.end_year is not None else max(files_by_year)

    for block_start, block_end in year_blocks(start_year, end_year, args.years_per_file):
        process_block(files_by_year, block_start, block_end, args)


if __name__ == "__main__":
    main()
