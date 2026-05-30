#!/usr/bin/env python3
"""Calculate ERA5 monthly sea ice concentration climatology."""

import argparse
import os
from glob import glob
from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar


EXPECTED_VAR = "sic_climatology"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ERA5 monthly sea ice concentration climatology."
    )
    parser.add_argument(
        "--input-glob",
        default="/g/data/rt52/era5/single-levels/reanalysis/ci/*/ci_era5_oper_sfc_*.nc",
        help="Glob for ERA5 sea ice concentration files.",
    )
    parser.add_argument(
        "--outfile",
        default="/g/data/ps29/nd0349/datasets/ERA5/era5_sic_monthly_climatology.nc",
        help="Output NetCDF path.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=24,
        help="Dask chunk size along time.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of Dask worker threads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing complete output file.",
    )
    return parser.parse_args()


def existing_file_is_complete(outfile):
    if not outfile.exists():
        return False

    with xr.open_dataset(outfile) as ds_existing:
        has_var = EXPECTED_VAR in ds_existing.data_vars
        has_month = "month" in ds_existing.coords and ds_existing.sizes.get("month") == 12

        if not has_var or not has_month:
            print(f"Existing file is incomplete or outdated: {outfile}")
            print(ds_existing)
            return False

        print(f"File already exists and looks complete: {outfile}")
        print(ds_existing)
        return True


def open_era5(input_glob, time_chunk):
    era5_files = sorted(glob(input_glob))

    if len(era5_files) == 0:
        raise FileNotFoundError(f"No ERA5 sea ice concentration files found: {input_glob}")

    print(f"Found {len(era5_files)} ERA5 files")
    print(f"First file: {era5_files[0]}")
    print(f"Last file:  {era5_files[-1]}")

    ds_era = xr.open_mfdataset(
        era5_files,
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


def calculate_climatology(ds_era):
    sic = get_sic(ds_era).astype("float32")

    # Monthly means first, then climatology. This gives each year equal
    # weight within each calendar month.
    sic_monthly = sic.resample(time="MS").mean("time")
    sic_clim = sic_monthly.groupby("time.month").mean("time")
    sic_clim = sic_clim.rename(EXPECTED_VAR)

    ds_clim = sic_clim.to_dataset()
    ds_clim = ds_clim.assign_coords(month=list(range(1, 13)))

    ds_clim[EXPECTED_VAR].attrs = {
        "long_name": "ERA5 sea ice concentration monthly climatology",
        "units": sic.attrs.get("units", "1"),
        "description": "Monthly climatological mean sea ice concentration",
    }

    ds_clim["month"].attrs = {
        "long_name": "calendar month",
        "description": "1 = January, ..., 12 = December",
    }

    ds_clim.attrs["source"] = "ERA5 single-levels sea ice concentration"
    ds_clim.attrs["method"] = (
        "Monthly means were calculated first, then averaged by calendar month "
        "to produce monthly climatologies."
    )

    return ds_clim


def write_output(ds_clim, outfile):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    tmpfile = outfile.with_suffix(".tmp.nc")

    if tmpfile.exists():
        tmpfile.unlink()

    encoding = {
        EXPECTED_VAR: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, 721, 1440),
        }
    }

    print(ds_clim)

    with ProgressBar():
        ds_clim.to_netcdf(tmpfile, encoding=encoding)

    tmpfile.replace(outfile)

    print(f"Saved: {outfile}")

    with xr.open_dataset(outfile) as ds_check:
        print(ds_check)
        print("Variables:", list(ds_check.data_vars))


def main():
    args = parse_args()
    outfile = Path(args.outfile)

    print("Running ERA5 SIC monthly climatology calculation")
    print(f"Output file: {outfile}")
    print(f"Input glob: {args.input_glob}")
    print(f"Dask worker threads: {args.n_workers}")
    print(f"Time chunk: {args.time_chunk}")

    if outfile.exists():
        if args.overwrite:
            print(f"Overwriting existing file: {outfile}")
            outfile.unlink()
        elif existing_file_is_complete(outfile):
            return
        else:
            print(f"Removing incomplete output file: {outfile}")
            outfile.unlink()

    dask.config.set(scheduler="threads", num_workers=args.n_workers)

    ds_era = open_era5(args.input_glob, args.time_chunk)
    ds_clim = calculate_climatology(ds_era)
    write_output(ds_clim, outfile)


if __name__ == "__main__":
    main()