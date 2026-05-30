#!/usr/bin/env python3
"""Calculate daily ASMR2 Antarctic sea ice area and extent by decade."""

import argparse
import gc
import os
import re
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


EXPECTED_VARS = {
    "antarctic_sie",
    "antarctic_sia",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate daily ASMR2 Antarctic sea ice area and extent time series, "
            "processing and saving one year block at a time."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/g/data/ps29/nd0349/datasets/uni-bremen/ASMR2/antarctic",
        help="Directory containing ASMR2 Antarctic NetCDF files.",
    )
    parser.add_argument(
        "--outdir",
        default="/g/data/ps29/nd0349/datasets/uni-bremen/ASMR2/processed",
        help="Directory for output NetCDF files.",
    )
    parser.add_argument(
        "--prefix",
        default="asmr2_antarctic_sea_ice_area_extent_daily",
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
        "--sic-var",
        default="z",
        help="ASMR2 sea ice concentration variable name.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Sea ice concentration threshold for extent.",
    )
    parser.add_argument(
        "--percent-scale",
        action="store_true",
        help="Treat SIC values as percent and divide by 100.",
    )
    parser.add_argument(
        "--x-dim",
        default=None,
        help="X dimension name. Default is inferred.",
    )
    parser.add_argument(
        "--y-dim",
        default=None,
        help="Y dimension name. Default is inferred.",
    )
    parser.add_argument(
        "--cell-area-km2",
        type=float,
        default=3.125 * 3.125,
        help=(
            "Grid-cell area in km2. ASMR2 is commonly distributed on a 3.125 km "
            "polar stereographic grid."
        ),
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
        help="Overwrite existing complete output files.",
    )
    return parser.parse_args()


def date_from_filename(path):
    match = re.search(r"(\d{8})", path.name)
    if match is None:
        raise ValueError(f"No YYYYMMDD date found in {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def file_year(path):
    return int(date_from_filename(path).year)


def find_files(input_dir):
    input_path = Path(input_dir)
    files = sorted(input_path.rglob("*.nc"))

    if not files:
        raise FileNotFoundError(f"No ASMR2 NetCDF files found under: {input_path}")

    files_by_year = {}
    for path in files:
        year = file_year(path)
        files_by_year.setdefault(year, []).append(path)

    print(f"Found {len(files)} ASMR2 files")
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


def block_outfile(outdir, prefix, start_year, end_year):
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


def open_asmr2(files, time_chunk, sic_var):
    file_strings = [str(path) for path in files]
    times = [date_from_filename(path) for path in files]

    def keep_sic(ds):
        if sic_var not in ds:
            raise ValueError(
                f"No ASMR2 SIC variable named {sic_var!r}. "
                f"Variables are: {list(ds.data_vars)}"
            )
        return ds[[sic_var]]

    ds_asmr2 = xr.open_mfdataset(
        file_strings,
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


def infer_xy_dims(sic, x_dim, y_dim):
    dims = [dim for dim in sic.dims if dim != "time"]

    if x_dim is not None and y_dim is not None:
        return x_dim, y_dim

    x_candidates = [dim for dim in dims if dim.lower() in {"x", "xc", "lon", "longitude"}]
    y_candidates = [dim for dim in dims if dim.lower() in {"y", "yc", "lat", "latitude"}]

    if x_dim is None and x_candidates:
        x_dim = x_candidates[0]
    if y_dim is None and y_candidates:
        y_dim = y_candidates[0]

    if x_dim is None or y_dim is None:
        if len(dims) != 2:
            raise ValueError(
                "Could not infer ASMR2 horizontal dimensions. "
                f"SIC dims are {sic.dims}; pass --x-dim and --y-dim."
            )
        y_dim = y_dim or dims[0]
        x_dim = x_dim or dims[1]

    return x_dim, y_dim


def calculate_daily_area_extent(ds_asmr2, args):
    sic = get_sic(ds_asmr2, args.sic_var, args.percent_scale)
    x_dim, y_dim = infer_xy_dims(sic, args.x_dim, args.y_dim)
    ice = sic > args.threshold

    cell_area = xr.ones_like(sic.isel(time=0), dtype="float32") * np.float32(
        args.cell_area_km2
    )
    cell_area.name = "cell_area"
    cell_area.attrs["units"] = "km2"

    ds_daily = xr.Dataset(
        {
            "antarctic_sie": cell_area.where(ice, 0).sum([y_dim, x_dim]),
            "antarctic_sia": (sic * cell_area).where(ice, 0).sum([y_dim, x_dim]),
        }
    )

    ds_daily["antarctic_sie"].attrs = {
        "long_name": "Daily Antarctic sea ice extent",
        "units": "km2",
        "description": (
            f"Southern Hemisphere grid-cell area where SIC > {args.threshold}"
        ),
    }
    ds_daily["antarctic_sia"].attrs = {
        "long_name": "Daily Antarctic sea ice area",
        "units": "km2",
        "description": (
            f"SIC-weighted Southern Hemisphere grid-cell area where SIC > "
            f"{args.threshold}"
        ),
    }

    ds_daily.attrs["source"] = "ASMR2 sea ice concentration, University of Bremen"
    ds_daily.attrs["threshold"] = f"SIC > {args.threshold}"
    ds_daily.attrs["cell_area_km2"] = args.cell_area_km2
    ds_daily.attrs["area_units"] = "km2"
    ds_daily.attrs["method"] = (
        "Daily sea ice area and extent were calculated from ASMR2 daily grids. "
        "Extent is the sum of grid-cell area where SIC exceeds the threshold; "
        "area is the sum of SIC-weighted grid-cell area over the same mask."
    )
    return ds_daily


def write_output(ds_out, outfile):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    tmpfile = outfile.with_suffix(".tmp.nc")

    if tmpfile.exists():
        tmpfile.unlink()

    time_chunksize = min(max(ds_out.sizes.get("time", 1), 1), 366)
    encoding = {
        var: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (time_chunksize,),
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
    outfile = block_outfile(args.outdir, args.prefix, start_year, end_year)

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

    files = sorted(files, key=date_from_filename)
    print(f"Processing {start_year}-{end_year} using {len(files)} files")

    ds_asmr2 = open_asmr2(files, args.time_chunk, args.sic_var)
    ds_asmr2 = ds_asmr2.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    ds_out = calculate_daily_area_extent(ds_asmr2, args)
    write_output(ds_out, outfile)

    ds_asmr2.close()
    ds_out.close()
    gc.collect()


def main():
    args = parse_args()

    print("Running ASMR2 daily Antarctic sea ice area/extent calculation")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.outdir}")
    print(f"Output prefix: {args.prefix}")
    print(f"SIC variable: {args.sic_var}")
    print(f"Threshold: {args.threshold}")
    print(f"Cell area: {args.cell_area_km2} km2")
    print(f"Dask worker threads: {args.n_workers}")
    print(f"Time chunk: {args.time_chunk}")
    print(f"Years per file: {args.years_per_file}")

    dask.config.set(scheduler="threads", num_workers=args.n_workers)

    files_by_year = find_files(args.input_dir)
    start_year = args.start_year if args.start_year is not None else min(files_by_year)
    end_year = args.end_year if args.end_year is not None else max(files_by_year)

    for block_start, block_end in year_blocks(start_year, end_year, args.years_per_file):
        process_block(files_by_year, block_start, block_end, args)


if __name__ == "__main__":
    main()
