#!/usr/bin/env python

from glob import glob
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# Add your ReadInAltika import here if needed, e.g.
# from your_module import ReadInAltika

def ReadInAltika(version, year=2019):
    if version == '0.6':
        month_range = range(1,13)
        df = pd.concat((pd.read_csv('/g/data/ps29/nd0349/Fraser-2024/data/v0_15/' + str(year) + ("%02d" % (month,)) + '_output_v0_6.csv') 
                        for month in tqdm(month_range, total = len(month_range), desc = "Reading in Alex's data")),
                        ignore_index=True)
        
        df = df[['first_meas_time', 'swhAtMyEdge', 'lonAtMyEdge', 'latAtMyEdge', 'lonAtAltiKaEdge','latAtAltiKaEdge', 'lonAtInnerMIZ', 'latAtInnerMIZ', 'mizWidthAlongTrackFromMyEdge', 'mizWidthAlongTrackFromAltikaEdge']]
        
    
    elif version == '0.10':
        # Version 0.10
        df_raw = pd.read_csv('data/v0_10/' + str(year) + '_all_output_v0_10.csv')
        row_temp = df_raw.loc[0,:]
        row_temp.values
        # Fill the first row with temporary data
        df = pd.DataFrame([row_temp], columns = df_raw.columns)
        
        numberRows,NumberCols = df_raw.shape
        
        for i in range(numberRows):
            row_temp = df_raw.loc[i,:]
            row = pd.DataFrame([row_temp], columns = df_raw.columns)
            if  (row['too_many_switches_flag'].values == 0) & (row['hit_continent_flag'].values == 0) & (row['ice_edge_diff_flag'].values == 0) & (row['latAtInnerMIZ'].values < row['latAtAltiKaEdge'].values) & (row['latAtInnerMIZ'].values < row['latAtMyEdge'].values) & (np.abs(row['lonAtInnerMIZ'].values - row['lonAtAltiKaEdge'].values) < 10):
                df = pd.concat([df,row])
        df.drop([0])
        df = df[['first_meas_time', 'swhAtMyEdge', 'lonAtMyEdge', 'latAtMyEdge', 'lonAtAltiKaEdge','latAtAltiKaEdge', 'lonAtInnerMIZ', 'latAtInnerMIZ', 'mizWidthAlongTrackFromMyEdge', 'mizWidthAlongTrackFromAltikaEdge']]
    elif version == '0.11':
        # Version 0.11
        df_raw = pd.read_csv('data/v0_11/' + str(year) + '_all_output_v0_11.csv')
        row_temp = df_raw.loc[0,:]
        row_temp.values
        # Fill the first row with temporary data
        df = pd.DataFrame([row_temp], columns = df_raw.columns)
        
        numberRows,NumberCols = df_raw.shape
        
        for i in range(numberRows):
            row_temp = df_raw.loc[i,:]
            row = pd.DataFrame([row_temp], columns = df_raw.columns)
            if  (row['too_many_switches_flag'].values == 0) & (row['hit_continent_flag'].values == 0) & (row['ice_edge_diff_flag'].values == 0) & (row['latAtInnerMIZ'].values < row['latAtAltiKaEdge'].values) & (row['latAtInnerMIZ'].values < row['latAtMyEdge'].values) & (np.abs(row['lonAtInnerMIZ'].values - row['lonAtAltiKaEdge'].values) < 10):
                df = pd.concat([df,row])
        df.drop([0])
        df = df[['first_meas_time', 'swhAtMyEdge', 'lonAtMyEdge', 'latAtMyEdge', 'lonAtAltiKaEdge','latAtAltiKaEdge', 'lonAtInnerMIZ', 'latAtInnerMIZ', 'mizWidthAlongTrackFromMyEdge', 'mizWidthAlongTrackFromAltikaEdge']]
    elif version == '0.12':
        # Version 0.12
        df_raw = pd.read_csv('data/v0_12/' + str(year) + '_all_output_v0_12.csv')
        row_temp = df_raw.loc[0,:]
        row_temp.values
        # Fill the first row with temporary data
        df = pd.DataFrame([row_temp], columns = df_raw.columns)
        
        numberRows,NumberCols = df_raw.shape
        
        for i in range(numberRows):
            row_temp = df_raw.loc[i,:]
            row = pd.DataFrame([row_temp], columns = df_raw.columns)
            if  (row['too_many_switches_flag'].values == 0) & (row['hit_continent_flag'].values == 0) & (row['ice_edge_diff_flag'].values == 0) & (row['latAtInnerMIZ'].values < row['latAtAltiKaEdge'].values) & (row['latAtInnerMIZ'].values < row['latAtMyEdge'].values) & (np.abs(row['lonAtInnerMIZ'].values - row['lonAtAltiKaEdge'].values) < 10):
                df = pd.concat([df,row])
        df.drop([0])
        df = df[['first_meas_time', 'swhAtMyEdge', 'lonAtMyEdge', 'latAtMyEdge', 'lonAtAltiKaEdge','latAtAltiKaEdge', 'lonAtInnerMIZ', 'latAtInnerMIZ', 'mizWidthAlongTrackFromMyEdge', 'mizWidthAlongTrackFromAltikaEdge']]
        
    elif version == '0.15':
        # Version 0.15
        df_raw = pd.read_csv('/g/data/ps29/nd0349/Fraser-2024/data/v0_15/' + str(year) + '_all_output_v0_15.csv')
        row_temp = df_raw.loc[0,:]
        row_temp.values
        # Fill the first row with temporary data
        df = pd.DataFrame([row_temp], columns = df_raw.columns)
        
        numberRows,NumberCols = df_raw.shape
        
        for i in range(numberRows):
            row_temp = df_raw.loc[i,:]
            row = pd.DataFrame([row_temp], columns = df_raw.columns)
            if  (row['too_many_switches_flag'].values == 0) & (row['hit_continent_flag'].values == 0) & (row['ice_edge_diff_flag'].values == 0) & (row['latAtInnerMIZ'].values < row['latAtAltiKaEdge'].values) & (row['latAtInnerMIZ'].values < row['latAtMyEdge'].values) & (np.abs(row['lonAtInnerMIZ'].values - row['lonAtAltiKaEdge'].values) < 10):
                df = pd.concat([df,row])
        df.drop([0])
        
    # Add dates to dataframe
    df['date'] = pd.to_datetime(df["first_meas_time"])#, format='%Y-%m-%d').dt.round("d")
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    return df

def haversine_km(lon1, lat1, lon2, lat2, radius=6371.0):
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


def lon_to_180(lon):
    return ((lon + 180) % 360) - 180


def read_altika_year(year, version="0.15"):
    df = pd.DataFrame(ReadInAltika(version=version, year=year))

    keep_cols = [
        "first_meas_time",
        "swhAtMyEdge",
        "lonAtMyEdge",
        "latAtMyEdge",
        "lonAtAltiKaEdge",
        "latAtAltiKaEdge",
        "lonAtInnerMIZ",
        "latAtInnerMIZ",
        "mizWidthAlongTrackFromMyEdge",
        "mizWidthAlongTrackFromAltikaEdge",
    ]

    df = df[keep_cols].copy()

    df["first_meas_time"] = pd.to_datetime(df["first_meas_time"])
    df["date"] = df["first_meas_time"].dt.date
    df["day"] = df["first_meas_time"].dt.day
    df["year"] = df["first_meas_time"].dt.year
    df["month"] = df["first_meas_time"].dt.month
    df["mizwidth_lat"] = (
        abs(df["latAtAltiKaEdge"] - df["latAtInnerMIZ"]) * 111.32
    )

    return df


def open_era5_siconc_month(year, month):
    files = sorted(glob(
        f"/g/data/rt52/era5/single-levels/reanalysis/ci/{year}/"
        f"ci_era5_oper_sfc_{year}{month:02d}*.nc"
    ))

    if len(files) == 0:
        raise FileNotFoundError(f"No ERA5 siconc file found for {year}-{month:02d}")

    return xr.open_dataset(files[0], engine="netcdf4")


def make_siconc_interpolator_from_da(siconc_da):
    era_lats = siconc_da["latitude"].values
    era_lons = siconc_da["longitude"].values
    era_vals = siconc_da.values

    if era_lats[0] > era_lats[-1]:
        era_lats = era_lats[::-1]
        era_vals = era_vals[::-1, :]

    return RegularGridInterpolator(
        (era_lats, era_lons),
        era_vals,
        bounds_error=False,
        fill_value=np.nan,
    )


def mean_siconc_along_track(row, siconc_interp, start="altika", n_points=50):
    if start == "altika":
        lon0 = row["lonAtAltiKaEdge"]
        lat0 = row["latAtAltiKaEdge"]
    elif start == "myedge":
        lon0 = row["lonAtMyEdge"]
        lat0 = row["latAtMyEdge"]
    else:
        raise ValueError("start must be 'altika' or 'myedge'")

    lon1 = row["lonAtInnerMIZ"]
    lat1 = row["latAtInnerMIZ"]

    lon0 = lon_to_180(lon0)
    lon1 = lon_to_180(lon1)

    if abs(lon1 - lon0) > 180:
        if lon0 > lon1:
            lon1 += 360
        else:
            lon0 += 360

    track_lons = np.linspace(lon0, lon1, n_points)
    track_lats = np.linspace(lat0, lat1, n_points)
    track_lons = lon_to_180(track_lons)

    points = np.column_stack([track_lats, track_lons])
    siconc_track = siconc_interp(points)

    return np.nanmean(siconc_track)


def add_era5_effective_miz_for_month(df_all, year, month, n_points=50):
    ds_era5 = open_era5_siconc_month(year, month)

    df_month = df_all[
        (df_all["first_meas_time"].dt.year == year)
        & (df_all["first_meas_time"].dt.month == month)
    ].copy()

    if len(df_month) == 0:
        ds_era5.close()
        return df_month

    df_month["altika_myedge_dist_km"] = haversine_km(
        df_month["lonAtAltiKaEdge"],
        df_month["latAtAltiKaEdge"],
        df_month["lonAtMyEdge"],
        df_month["latAtMyEdge"],
    )

    df_month = df_month[df_month["altika_myedge_dist_km"] < 50].copy()
    df_month = df_month[df_month["mizWidthAlongTrackFromAltikaEdge"] < 500].copy()

    if len(df_month) == 0:
        ds_era5.close()
        return df_month

    era_times = pd.to_datetime(ds_era5["time"].values)

    nearest_idx = era_times.get_indexer(
        pd.to_datetime(df_month["first_meas_time"]),
        method="nearest",
    )

    df_month["era5_time"] = era_times[nearest_idx]
    df_month["era5_time_idx"] = nearest_idx

    df_month["mean_siconc_altika_track"] = np.nan
    df_month["mean_siconc_myedge_track"] = np.nan

    for time_idx, group in tqdm(
        df_month.groupby("era5_time_idx"),
        desc=f"ERA5 siconc {year}-{month:02d}"
    ):
        siconc_da = ds_era5["siconc"].isel(time=int(time_idx))
        siconc_interp = make_siconc_interpolator_from_da(siconc_da)

        for idx, row in group.iterrows():
            df_month.loc[idx, "mean_siconc_altika_track"] = mean_siconc_along_track(
                row,
                siconc_interp=siconc_interp,
                start="altika",
                n_points=n_points
            )

            df_month.loc[idx, "mean_siconc_myedge_track"] = mean_siconc_along_track(
                row,
                siconc_interp=siconc_interp,
                start="myedge",
                n_points=n_points
            )

    df_month["mizWidthAlongTrackFromAltikaEdge_eff_era5"] = (
        df_month["mizWidthAlongTrackFromAltikaEdge"]
        * df_month["mean_siconc_altika_track"]
    )

    df_month["mizWidthAlongTrackFromMyEdge_eff_era5"] = (
        df_month["mizWidthAlongTrackFromMyEdge"]
        * df_month["mean_siconc_myedge_track"]
    )

    ds_era5.close()
    return df_month


def process_year(year, n_points, version, out_dir):
    print(f"Reading AltiKa for {year}")
    df_all = read_altika_year(year, version=version)

    monthly_results = []

    for period in tqdm(pd.period_range(f"{year}-01", f"{year}-12", freq="M")):
        try:
            df_month_eff = add_era5_effective_miz_for_month(
                df_all,
                year=period.year,
                month=period.month,
                n_points=n_points,
            )
            monthly_results.append(df_month_eff)

        except FileNotFoundError as e:
            print(e)

    if len(monthly_results) == 0:
        print(f"No output for {year}")
        return

    df_all_eff = pd.concat(monthly_results, ignore_index=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{year}_effective_hourly_{n_points}_v0_15.csv"
    df_all_eff.to_csv(out_file, index=False)

    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--version", type=str, default="0.15")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/g/data/ps29/nd0349/Fraser-2024/data/cleaned",
    )

    args = parser.parse_args()

    process_year(
        year=args.year,
        n_points=args.n_points,
        version=args.version,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

