#!/usr/bin/env python3
"""Calculate ASMR2 monthly Antarctic sea ice concentration climatology."""

import argparse
import os
import re
from pathlib import Path

import dask
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


EXPECTED_VAR = "sic_climatology"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ASMR2 monthly Antarctic sea ice concentration climatology."
    )
    parser.add_argument(
        "--input-dir",
        default="/g/data/ps29/nd0349/datasets/uni-bremen/ASMR2/antarctic",
        help="Directory containing ASMR2 Antarctic NetCDF files.",
    )
    parser.add_argument(
        "--outfile",
        default=(
            "/g/data/ps29/nd0349/datasets/uni-bremen/ASMR2/processed/"
            "asmr2_sic_monthly_climatology.nc"
        ),
        help="Output NetCDF path.",
    )
    parser.add_argument(
        "--sic-var",
        default="z",
        help="ASMR2 sea ice concentration variable name.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year to use. Default is inferred from input files.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to use. Default is inferred from input files.",
    )
    parser.add_argument(
        "--percent-scale",
        action="store_true",
        help="Treat SIC values as percent and divide by 100.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=31,
        help="Dask chunk size along time.",
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
        help="Overwrite an existing complete output file.",
    )
    return parser.parse_args()


def date_from_filename(path):
    match = re.search(r"(\d{8})", path.name)
    if match is None:
        raise ValueError(f"No YYYYMMDD date found in {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


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


def find_files(input_dir, start_year, end_year):
    input_path = Path(input_dir)
    files = sorted(input_path.rglob("*.nc"), key=date_from_filename)

    if not files:
        raise FileNotFoundError(f"No ASMR2 NetCDF files found under: {input_path}")

    dated_files = [(path, date_from_filename(path)) for path in files]
    if start_year is not None:
        dated_files = [(path, date) for path, date in dated_files if date.year >= start_year]
    if end_year is not None:
        dated_files = [(path, date) for path, date in dated_files if date.year <= end_year]

    if not dated_files:
        raise FileNotFoundError(
            "No ASMR2 NetCDF files remain after applying year filters."
        )

    files = [path for path, date in dated_files]
    times = [date for path, date in dated_files]

    print(f"Found {len(files)} ASMR2 files")
    print(f"First file: {files[0]}")
    print(f"Last file:  {files[-1]}")
    print(f"Years used: {times[0].year}-{times[-1].year}")
    return files, times


def open_asmr2(input_dir, start_year, end_year, time_chunk, sic_var):
    files, times = find_files(input_dir, start_year, end_year)

    def keep_sic(ds):
        if sic_var not in ds:
            raise ValueError(
                f"No ASMR2 SIC variable named {sic_var!r}. "
                f"Variables are: {list(ds.data_vars)}"
            )
        return ds[[sic_var]]

    ds_asmr2 = xr.open_mfdataset(
        [str(path) for path in files],
        combine="nested",
        concat_dim="time",
        decode_times=True,
        parallel=True,
        chunks={"time": time_chunk},
        preprocess=keep_sic,
        data_vars="all",
        coords="minimal",
        compat="override",
    )

    ds_asmr2 = ds_asmr2.assign_coords(time=pd.DatetimeIndex(times))
    return ds_asmr2.chunk({"time": time_chunk})


def get_sic(ds_asmr2, var_name, percent_scale):
    if var_name not in ds_asmr2:
        raise ValueError(
            f"No ASMR2 SIC variable named {var_name!r}. "
            f"Variables are: {list(ds_asmr2.data_vars)}"
        )

    sic = ds_asmr2[var_name].astype("float32")
    if percent_scale or float(sic.max(skipna=True).compute()) > 1.5:
        sic = sic / 100.0

    return sic.clip(min=0.0, max=1.0)


def calculate_climatology(ds_asmr2, sic_var, percent_scale):
    sic = get_sic(ds_asmr2, sic_var, percent_scale)

    # Monthly means first, then climatology. This gives each year equal
    # weight within each calendar month.
    sic_monthly = sic.resample(time="MS").mean("time")
    sic_clim = sic_monthly.groupby("time.month").mean("time")
    sic_clim = sic_clim.rename(EXPECTED_VAR)

    ds_clim = sic_clim.to_dataset()
    ds_clim = ds_clim.assign_coords(month=list(range(1, 13)))

    ds_clim[EXPECTED_VAR].attrs = {
        "long_name": "ASMR2 Antarctic sea ice concentration monthly climatology",
        "units": "1",
        "description": "Monthly climatological mean sea ice concentration",
    }

    ds_clim["month"].attrs = {
        "long_name": "calendar month",
        "description": "1 = January, ..., 12 = December",
    }

    ds_clim.attrs["source"] = "ASMR2 sea ice concentration, University of Bremen"
    ds_clim.attrs["method"] = (
        "Monthly means were calculated first, then averaged by calendar month "
        "to produce monthly climatologies."
    )

    return ds_clim


def climatology_chunksizes(ds_clim):
    chunksizes = []
    for dim in ds_clim[EXPECTED_VAR].dims:
        if dim == "month":
            chunksizes.append(1)
        else:
            chunksizes.append(ds_clim.sizes[dim])
    return tuple(chunksizes)


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
            "chunksizes": climatology_chunksizes(ds_clim),
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

    print("Running ASMR2 SIC monthly climatology calculation")
    print(f"Output file: {outfile}")
    print(f"Input directory: {args.input_dir}")
    print(f"SIC variable: {args.sic_var}")
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

    ds_asmr2 = open_asmr2(
        args.input_dir,
        args.start_year,
        args.end_year,
        args.time_chunk,
        args.sic_var,
    )
    ds_clim = calculate_climatology(ds_asmr2, args.sic_var, args.percent_scale)
    write_output(ds_clim, outfile)

    ds_asmr2.close()
    ds_clim.close()


if __name__ == "__main__":
    main()
