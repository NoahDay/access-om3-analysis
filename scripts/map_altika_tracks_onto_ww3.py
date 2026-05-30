#!/usr/bin/env python3

from pathlib import Path
import argparse

import intake
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm


ALTIKA_BASE = Path("/g/data/ps29/nd0349/Fraser-2024/data")

GRID_FILE = (
    "/g/data/vk83/configurations/inputs/access-om3/cice/grids/"
    "global.1deg/2024.05.14/grid.nc"
)

ALTIKA_COLS = [
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


def read_in_altika(version, year, altika_base=ALTIKA_BASE):
    if version == "0.6":
        files = [
            altika_base / "v0_15" / f"{year}{month:02d}_output_v0_6.csv"
            for month in range(1, 13)
        ]

        return pd.concat(
            (pd.read_csv(path) for path in tqdm(files, desc="Reading AltiKa")),
            ignore_index=True,
        )

    version_tag = version.replace(".", "_")
    path = altika_base / f"v{version_tag}" / f"{year}_all_output_v{version_tag}.csv"

    df_raw = pd.read_csv(path)

    good = (
        (df_raw["too_many_switches_flag"] == 0)
        & (df_raw["hit_continent_flag"] == 0)
        & (df_raw["ice_edge_diff_flag"] == 0)
        & (df_raw["latAtInnerMIZ"] < df_raw["latAtAltiKaEdge"])
        & (df_raw["latAtInnerMIZ"] < df_raw["latAtMyEdge"])
        & (np.abs(df_raw["lonAtInnerMIZ"] - df_raw["lonAtAltiKaEdge"]) < 10)
    )

    return df_raw.loc[good].copy()


def lon_to_180(lon):
    return ((lon + 180) % 360) - 180


def frequency_to_period(freq):
    freq = np.asarray(freq, dtype=float)
    return np.where(freq > 0, 1 / freq, np.nan)


def haversine_km(lon1, lat1, lon2, lat2, radius=6371.0):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return radius * 2 * np.arcsin(np.sqrt(a))


def prepare_altika_df(
    year,
    version="0.15",
    altika_base=ALTIKA_BASE,
    max_edge_distance_km=50,
    max_miz_width_km=500,
):
    df = read_in_altika(version=version, year=year, altika_base=altika_base)
    df = pd.DataFrame(df)[ALTIKA_COLS].copy()

    df["first_meas_time"] = pd.to_datetime(df["first_meas_time"])
    df["date"] = df["first_meas_time"].dt.date
    df["day"] = df["first_meas_time"].dt.day
    df["year"] = df["first_meas_time"].dt.year
    df["month"] = df["first_meas_time"].dt.month

    df["altika_myedge_dist_km"] = haversine_km(
        df["lonAtAltiKaEdge"],
        df["latAtAltiKaEdge"],
        df["lonAtMyEdge"],
        df["latAtMyEdge"],
    )

    df = df[df["altika_myedge_dist_km"] < max_edge_distance_km].copy()
    df = df[df["mizWidthAlongTrackFromAltikaEdge"] < max_miz_width_km].copy()

    return df


def open_ww3_dataset_from_intake(
    esm_file,
    output_frequency,
    grid_file=GRID_FILE,
):
    datastore = intake.open_esm_datastore(
        esm_file,
        columns_with_iterables=[
            "variable",
            "variable_long_name",
            "variable_standard_name",
            "variable_cell_methods",
            "variable_units",
        ],
    )

    variables = [
        "HS",
        "ICE",
        "ICEF",
        "ICEH",
        "THM",
        "FP0",
        "T02",
        "UAX",
        "UAY",
    ]

    search_kwargs = {
        "variable": variables,
        "require_all_on": "path",
    }

    if output_frequency is not None:
        search_kwargs["frequency"] = output_frequency

    ds_ww3 = datastore.search(**search_kwargs).to_dask(
        xarray_open_kwargs={"chunks": {"time": 1}}
    )

    if "ni" in ds_ww3.dims or "nj" in ds_ww3.dims:
        ds_ww3 = ds_ww3.rename({"ni": "nx", "nj": "ny"})

    grid_ds = xr.open_dataset(grid_file)

    ds_ww3 = ds_ww3.assign_coords(
        TLON=(("ny", "nx"), np.degrees(grid_ds["tlon"].values)),
        TLAT=(("ny", "nx"), np.degrees(grid_ds["tlat"].values)),
    )

    ds_ww3["tarea"] = (("ny", "nx"), grid_ds["tarea"].values)
    ds_ww3["HTE"] = (("ny", "nx"), grid_ds["hte"].values / 100)

    grid_ds.close()

    return ds_ww3


def subset_ww3_for_mapping(
    ds_ww3,
    year,
    ice_var="ICE",
    edge_vars=None,
    period_vars=None,
    freq_to_period_vars=None,
    lon_name="TLON",
    lat_name="TLAT",
    time_name="time",
):
    edge_vars = edge_vars or {"HS": "HS", "THM": "THM", "UAX": "UAX", "UAY": "UAY"}
    period_vars = period_vars or {"T02": "TM02"}
    freq_to_period_vars = freq_to_period_vars or {"FP0": "TP"}

    needed_vars = (
        [ice_var, lon_name, lat_name]
        + list(edge_vars.keys())
        + list(period_vars.keys())
        + list(freq_to_period_vars.keys())
    )

    needed_vars = list(dict.fromkeys(needed_vars))

    return ds_ww3[needed_vars].sel(
        {time_name: slice(f"{year}-01-01", f"{year}-12-31")}
    )


def build_ww3_grid_lookup(ds_ww3, lon_name="TLON", lat_name="TLAT", max_lat=-40):
    lon = ds_ww3[lon_name].values
    lat = ds_ww3[lat_name].values

    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("Expected 2D WW3 TLON/TLAT arrays")

    southern = lat <= max_lat
    y_all, x_all = np.where(southern)

    lon_rad = np.deg2rad(lon_to_180(lon[southern]))
    lat_rad = np.deg2rad(lat[southern])

    xyz = np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )

    return {
        "tree": cKDTree(xyz),
        "y_all": y_all,
        "x_all": x_all,
        "lats": lat,
    }


def make_track_points(
    row,
    start="altika",
    n_points=200,
    extend_factor=5.0,
):
    if start == "altika":
        lon0 = row["lonAtAltiKaEdge"]
        lat0 = row["latAtAltiKaEdge"]
    elif start == "myedge":
        lon0 = row["lonAtMyEdge"]
        lat0 = row["latAtMyEdge"]
    else:
        raise ValueError("start must be 'altika' or 'myedge'")

    lon_inner = row["lonAtInnerMIZ"]
    lat_inner = row["latAtInnerMIZ"]

    lon0 = lon_to_180(lon0)
    lon_inner = lon_to_180(lon_inner)

    if abs(lon_inner - lon0) > 180:
        if lon0 > lon_inner:
            lon_inner += 360
        else:
            lon0 += 360

    lon_end = lon0 + extend_factor * (lon_inner - lon0)
    lat_end = lat0 + extend_factor * (lat_inner - lat0)

    lons = lon_to_180(np.linspace(lon0, lon_end, n_points))
    lats = np.linspace(lat0, lat_end, n_points)

    return lons, lats


def nearest_grid_indices(lons, lats, grid):
    lon_rad = np.deg2rad(lon_to_180(np.asarray(lons)))
    lat_rad = np.deg2rad(np.asarray(lats))

    xyz = np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )

    _, southern_idx = grid["tree"].query(xyz)

    return grid["y_all"][southern_idx], grid["x_all"][southern_idx]

def one_cell_north(y, x, grid_lats):
    y = int(y)
    x = int(x)

    candidates = []

    if y > 0:
        candidates.append((y - 1, x))

    if y < grid_lats.shape[0] - 1:
        candidates.append((y + 1, x))

    if len(candidates) == 0:
        return y, x

    lat_here = grid_lats[y, x]

    north_y, north_x = max(
        candidates,
        key=lambda ij: grid_lats[ij[0], ij[1]] - lat_here,
    )

    if grid_lats[north_y, north_x] > lat_here:
        return north_y, north_x

    return y, x

def unique_grid_track(y_idx, x_idx):
    pairs = np.column_stack([y_idx, x_idx])
    _, first = np.unique(pairs, axis=0, return_index=True)
    first = np.sort(first)

    return y_idx[first], x_idx[first]


def add_track_output_columns(
    df,
    suffix,
    ice_var,
    edge_vars,
    period_vars,
    freq_to_period_vars,
):
    output_cols = [
        f"ww3_edge_y_{suffix}",
        f"ww3_edge_x_{suffix}",
        f"ww3_edge_north_y_{suffix}",
        f"ww3_edge_north_x_{suffix}",
        f"ww3_inner_y_{suffix}",
        f"ww3_inner_x_{suffix}",
        f"ww3_track_grid_cell_count_{suffix}",
        f"ww3_miz_width_hs05_km_{suffix}",
        f"ww3_miz_width_hs05_found_{suffix}",
        f"{ice_var}_track_mean_{suffix}",
        f"{ice_var}_track_min_{suffix}",
        f"{ice_var}_track_max_{suffix}",
        f"{ice_var}_edge_{suffix}",
        f"{ice_var}_edge_north_{suffix}",
        f"{ice_var}_inner_{suffix}",
    ]

    for out_name in edge_vars.values():
        output_cols += [
            f"{out_name}_edge_{suffix}",
            f"{out_name}_edge_north_{suffix}",
            f"{out_name}_inner_{suffix}",
        ]

    for out_name in period_vars.values():
        output_cols += [
            f"{out_name}_edge_{suffix}",
            f"{out_name}_edge_north_{suffix}",
            f"{out_name}_inner_{suffix}",
        ]

    for out_name in freq_to_period_vars.values():
        output_cols += [
            f"{out_name}_edge_{suffix}",
            f"{out_name}_edge_north_{suffix}",
            f"{out_name}_inner_{suffix}",
        ]
    for col in output_cols:
        df[col] = np.nan

    return df


def fill_track_outputs(
    df,
    idx,
    suffix,
    fields,
    ice_var,
    edge_vars,
    period_vars,
    freq_to_period_vars,
    y_idx,
    x_idx,
    ww3_miz_width_hs05_km=np.nan,
    ww3_miz_width_hs05_found=0,
):
    edge_y, edge_x = y_idx[0], x_idx[0]
    inner_y, inner_x = y_idx[-1], x_idx[-1]
    edge_north_y, edge_north_x = one_cell_north(
        edge_y,
        edge_x,
        fields["_TLAT"],
    )

    ice_track = fields[ice_var][y_idx, x_idx]

    df.loc[idx, f"ww3_edge_y_{suffix}"] = edge_y
    df.loc[idx, f"ww3_edge_x_{suffix}"] = edge_x
    df.loc[idx, f"ww3_edge_north_y_{suffix}"] = edge_north_y
    df.loc[idx, f"ww3_edge_north_x_{suffix}"] = edge_north_x
    df.loc[idx, f"ww3_inner_y_{suffix}"] = inner_y
    df.loc[idx, f"ww3_inner_x_{suffix}"] = inner_x
    df.loc[idx, f"ww3_track_grid_cell_count_{suffix}"] = len(y_idx)

    df.loc[idx, f"{ice_var}_track_mean_{suffix}"] = np.nanmean(ice_track)
    df.loc[idx, f"{ice_var}_track_min_{suffix}"] = np.nanmin(ice_track)
    df.loc[idx, f"{ice_var}_track_max_{suffix}"] = np.nanmax(ice_track)
    df.loc[idx, f"{ice_var}_edge_{suffix}"] = fields[ice_var][edge_y, edge_x]
    df.loc[idx, f"{ice_var}_inner_{suffix}"] = fields[ice_var][inner_y, inner_x]
    df.loc[idx, f"{ice_var}_edge_north_{suffix}"] = fields[ice_var][
        edge_north_y, edge_north_x
    ]

    df.loc[idx, f"ww3_miz_width_hs05_km_{suffix}"] = ww3_miz_width_hs05_km
    df.loc[idx, f"ww3_miz_width_hs05_found_{suffix}"] = ww3_miz_width_hs05_found
    df.loc[idx, f"ww3_edge_north_y_{suffix}"] = edge_north_y
    df.loc[idx, f"ww3_edge_north_x_{suffix}"] = edge_north_x

    for in_var, out_name in edge_vars.items():
        df.loc[idx, f"{out_name}_edge_{suffix}"] = fields[in_var][edge_y, edge_x]
        df.loc[idx, f"{out_name}_edge_north_{suffix}"] = fields[in_var][edge_north_y, edge_north_x]
        df.loc[idx, f"{out_name}_inner_{suffix}"] = fields[in_var][inner_y, inner_x]

    for in_var, out_name in period_vars.items():
        df.loc[idx, f"{out_name}_edge_{suffix}"] = fields[in_var][edge_y, edge_x]
        df.loc[idx, f"{out_name}_edge_north_{suffix}"] = fields[in_var][
            edge_north_y, edge_north_x
        ]
        df.loc[idx, f"{out_name}_inner_{suffix}"] = fields[in_var][inner_y, inner_x]

    for freq_var, out_name in freq_to_period_vars.items():
        fp_edge = fields[freq_var][edge_y, edge_x]
        fp_edge_north = fields[freq_var][edge_north_y, edge_north_x]
        fp_inner = fields[freq_var][inner_y, inner_x]

        df.loc[idx, f"{out_name}_edge_{suffix}"] = frequency_to_period(fp_edge)
        df.loc[idx, f"{out_name}_edge_north_{suffix}"] = frequency_to_period(
            fp_edge_north
        )
        df.loc[idx, f"{out_name}_inner_{suffix}"] = frequency_to_period(fp_inner)


def cumulative_track_distance_km(lons, lats):
    lons = np.asarray(lons)
    lats = np.asarray(lats)

    dist = np.zeros(len(lons))

    if len(lons) <= 1:
        return dist

    step_dist = haversine_km(
        lons[:-1],
        lats[:-1],
        lons[1:],
        lats[1:],
    )

    dist[1:] = np.cumsum(step_dist)

    return dist


def miz_width_from_hs_threshold(lons, lats, hs_values, threshold=0.5):
    hs_values = np.asarray(hs_values, dtype=float)
    dist_km = cumulative_track_distance_km(lons, lats)

    below = np.where(hs_values < threshold)[0]

    if len(below) == 0:
        return np.nan, 0

    first = below[0]

    if first == 0:
        return 0.0, 1

    hs0 = hs_values[first - 1]
    hs1 = hs_values[first]
    d0 = dist_km[first - 1]
    d1 = dist_km[first]

    if np.isfinite(hs0) and np.isfinite(hs1) and hs1 != hs0:
        frac = (threshold - hs0) / (hs1 - hs0)
        width_km = d0 + frac * (d1 - d0)
    else:
        width_km = d1

    return width_km, 1


def map_altika_tracks_to_ww3_both_edges_fast(
    df_altika,
    ds_ww3,
    year,
    ice_var="ICE",
    edge_vars=None,
    period_vars=None,
    freq_to_period_vars=None,
    lon_name="TLON",
    lat_name="TLAT",
    time_name="time",
    n_points=200,
    max_lat=-40,
    extend_factor=5.0,
):
    edge_vars = edge_vars or {"HS": "HS", "THM": "THM", "UAX": "UAX", "UAY": "UAY"}
    period_vars = period_vars or {"T02": "TM02"}
    freq_to_period_vars = freq_to_period_vars or {"FP0": "TP"}

    ds_year = subset_ww3_for_mapping(
        ds_ww3,
        year=year,
        ice_var=ice_var,
        edge_vars=edge_vars,
        period_vars=period_vars,
        freq_to_period_vars=freq_to_period_vars,
        lon_name=lon_name,
        lat_name=lat_name,
        time_name=time_name,
    )

    df = df_altika.copy()
    df["first_meas_time"] = pd.to_datetime(df["first_meas_time"])

    grid = build_ww3_grid_lookup(
        ds_year,
        lon_name=lon_name,
        lat_name=lat_name,
        max_lat=max_lat,
    )

    ww3_times = pd.DatetimeIndex(pd.to_datetime(ds_year[time_name].values))
    time_idx = ww3_times.get_indexer(df["first_meas_time"], method="nearest")

    if np.any(time_idx < 0):
        raise ValueError("Could not match every AltiKa time to a WW3 time")

    df["ww3_time"] = ww3_times[time_idx]
    df["ww3_time_idx"] = time_idx

    for suffix in ["altikaedge", "myedge"]:
        df = add_track_output_columns(
            df,
            suffix=suffix,
            ice_var=ice_var,
            edge_vars=edge_vars,
            period_vars=period_vars,
            freq_to_period_vars=freq_to_period_vars,
        )

    vars_to_load = set([ice_var])
    vars_to_load.update(edge_vars.keys())
    vars_to_load.update(period_vars.keys())
    vars_to_load.update(freq_to_period_vars.keys())

    for ti, group in tqdm(df.groupby("ww3_time_idx"), desc="Mapping tracks to WW3"):
        fields = {
            var: np.asarray(ds_year[var].isel({time_name: int(ti)}).values)
            for var in vars_to_load
        }
        fields["_TLAT"] = np.asarray(ds_year["TLAT"].values)

        for idx, row in group.iterrows():
            for start, suffix in [("altika", "altikaedge"), ("myedge", "myedge")]:
                lons, lats = make_track_points(
                    row,
                    start=start,
                    n_points=n_points,
                    extend_factor=extend_factor,
                )

                y_raw, x_raw = nearest_grid_indices(lons, lats, grid)

                hs_track = fields["HS"][y_raw, x_raw]

                ww3_width_km, ww3_width_found = miz_width_from_hs_threshold(
                    lons,
                    lats,
                    hs_track,
                    threshold=0.5,
                )

                y_idx, x_idx = unique_grid_track(y_raw, x_raw)

                fill_track_outputs(
                    df=df,
                    idx=idx,
                    suffix=suffix,
                    fields=fields,
                    ice_var=ice_var,
                    edge_vars=edge_vars,
                    period_vars=period_vars,
                    freq_to_period_vars=freq_to_period_vars,
                    y_idx=y_idx,
                    x_idx=x_idx,
                    ww3_miz_width_hs05_km=ww3_width_km,
                    ww3_miz_width_hs05_found=ww3_width_found,
                )

    return df


def process_year(args):
    print(f"Processing year {args.year}")

    print("Opening WW3 datastore")
    ds_ww3 = open_ww3_dataset_from_intake(
        esm_file=args.esm_file,
        output_frequency=args.output_frequency,
        grid_file=args.grid_file,
    )

    print("Reading AltiKa tracks")
    df_altika = prepare_altika_df(
        year=args.year,
        version=args.version,
        altika_base=Path(args.altika_base),
        max_edge_distance_km=args.max_edge_distance_km,
        max_miz_width_km=args.max_miz_width_km,
    )

    print(f"AltiKa rows after filtering: {len(df_altika)}")

    if len(df_altika) == 0:
        print("No AltiKa rows after filtering; skipping output")
        return

    max_track_lat = df_altika[
        ["latAtAltiKaEdge", "latAtMyEdge", "latAtInnerMIZ"]
    ].max().max()

    if max_track_lat > args.max_lat:
        print(
            f"Warning: max AltiKa latitude is {max_track_lat:.2f}, "
            f"but max_lat is {args.max_lat:.2f}."
        )

    print("Mapping AltiKa tracks to WW3")
    df_tracks = map_altika_tracks_to_ww3_both_edges_fast(
        df_altika=df_altika,
        ds_ww3=ds_ww3,
        year=args.year,
        ice_var="ICE",
        edge_vars={"HS": "HS", "THM": "THM", "UAX": "UAX", "UAY": "UAY"},
        period_vars={"T02": "TM02"},
        freq_to_period_vars={"FP0": "TP"},
        lon_name="TLON",
        lat_name="TLAT",
        time_name="time",
        n_points=args.n_points,
        max_lat=args.max_lat,
        extend_factor=args.extend_factor,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / (
        f"{args.year}_altika_tracks_on_ww3_both_edges_"
        f"n{args.n_points}_v{args.version.replace('.', '_')}.csv"
    )

    df_tracks.to_csv(out_file, index=False)
    ds_ww3.close()

    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Map AltiKa MIZ tracks onto WW3 output."
    )

    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--esm-file", type=str, required=True)
    parser.add_argument("--output-frequency", type=str, default="1D")
    parser.add_argument("--grid-file", type=str, default=GRID_FILE)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/g/data/ps29/nd0349/Fraser-2024/data/cleaned",
    )

    parser.add_argument("--version", type=str, default="0.15")
    parser.add_argument("--altika-base", type=str, default=str(ALTIKA_BASE))

    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--max-lat", type=float, default=-40)

    parser.add_argument("--max-edge-distance-km", type=float, default=50)
    parser.add_argument("--max-miz-width-km", type=float, default=500)

    parser.add_argument("--extend-factor", type=float, default=10.0)

    args = parser.parse_args()
    process_year(args)


if __name__ == "__main__":
    main()