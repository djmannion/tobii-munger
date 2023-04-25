from pathlib import Path

import polars as pl
import numpy as np
import xarray as xr

__all__ = ["read_unified"]


def _name_generator(ii: int) -> str:
    """Generate column names for the `vals` column in the Tobii data."""
    return "xyz"[ii]


def _to_xarray(data: pl.DataFrame) -> dict:
    "Converts the Polars DataFrame from `read_unified(as_xarray=False)` to `xarray` format"

    def convert_timestamps(datatype):
        timestamps_s = data.filter(predicate=pl.col("type") == datatype).select(exprs=pl.col("timestamp"))
        timestamps_s = np.squeeze(np.array(timestamps_s))
        # convert this to nanoseconds for xarray
        timestamps_ns = (timestamps_s * 1e9).astype("timedelta64[ns]")
        assert np.allclose(timestamps_ns / np.timedelta64(1, "s"), timestamps_s)
        return timestamps_ns

    gaze_timestamps = convert_timestamps(datatype="gaze2d")

    # gaze is a dataset with dims time, camera_space, world_space, eye
    # dvs are: gaze_camera, gaze_world, pupil_diameter, gaze_origin, gaze_direction
    gaze_screen_data = data.filter(predicate=pl.col("type") == "gaze2d").select(pl.col("x", "y")).to_numpy()

    gaze_screen = xr.DataArray(
        data=gaze_screen_data,
        dims=("time", "camera_space"),
        coords={"time": gaze_time_ns, "camera_space": np.array(["x", "y"], dtype=str)},
    )

    return gaze_screen


    n_gaze_samples = len(gaze_time)



    return gaze_time



def read_unified(path: Path | str, datatype: str | None = None, as_xarray: bool = False) -> pl.DataFrame | dict:
    """Read Tobii data from a unified parquet file.

    Arguments:
        path: PathLike
            The path of the unified parquet file to read.
        datatype: str | None, optional
            The datatype to filter the data by. If None, all data is returned.

    Returns:
        pl.DataFrame
            A Polars DataFrame containing the Tobii data from the unified parquet file.
    """
    data = pl.scan_parquet(path)
    if datatype is not None:
        data = data.filter(pl.col("type") == datatype)
    data = data.with_columns(
        pl.col("vals").arr.to_struct(n_field_strategy="max_width", name_generator=_name_generator)
    ).unnest("vals")
    if datatype is not None:
        data = data.drop("type")
    data = data.collect()
    if as_xarray:
        data = _to_xarray(data=data)
    return data
