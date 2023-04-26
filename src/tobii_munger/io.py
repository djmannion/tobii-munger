from pathlib import Path

import polars as pl
import numpy as np
import xarray as xr

__all__ = ["read_unified"]


def _name_generator(ii: int) -> str:
    """Generate column names for the `vals` column in the Tobii data."""
    return "xyz"[ii]


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


def _to_xarray(data: pl.DataFrame) -> dict:
    "Converts the Polars DataFrame from `read_unified(as_xarray=False)` to `xarray` format"

    def convert_timestamps(datatype):
        timestamps_s = data.filter(predicate=pl.col("type") == datatype).select(exprs=pl.col("timestamp"))
        timestamps_s = np.squeeze(np.array(timestamps_s))
        # convert this to nanoseconds for xarray
        timestamps_ns = (timestamps_s * 1e9).astype("timedelta64[ns]")
        assert np.allclose(timestamps_ns / np.timedelta64(1, "s"), timestamps_s)
        return timestamps_ns

    # gaze sample data

    # dimensions and coordinates
    # each tuple is (dim_name, dim_coords, dim_attrs)
    # we will define the time dimension separately for each data array
    norm_space_dim = (
        "norm_space",
        np.array(["x", "y"], dtype=str),
        {"units": "Normalised video space; 0 is top/left, 1 is bottom/right"},
    )
    world_space_dim = (
        "world_space",
        np.array(["x", "y", "z"], dtype=str),
        {"units": "Relative to the scene camera, in mm"},
    )

    # form data arrays
    gaze_video = xr.DataArray(
        data=data.filter(predicate=pl.col("type") == "gaze2d").select(pl.col("x", "y")).to_numpy(),
        dims=("time", "norm_space"),
        coords=(
            ("time", convert_timestamps(datatype="gaze2d")),
            norm_space_dim,
        ),
        attrs={
            "info": "Position of gaze in the scene camera",
        },
    )

    gaze_world = xr.DataArray(
        data=data.filter(predicate=pl.col("type") == "gaze3d").select(pl.col("x", "y", "z")).to_numpy(),
        dims=("time", "world_space"),
        coords=(
            ("time", convert_timestamps(datatype="gaze3d")),
            world_space_dim,
        ),
        attrs={
            "info": "Position of the vergence point of the left and right gaze vector relative to the scene camera.",
        },
    )

    # these are per-eye, so need to iterate over eyes
    gaze_origin = []
    gaze_direction = []
    pupil_diameter = []

    for eye in ["left", "right"]:

        gaze_origin.append(
            xr.DataArray(
                data=data.filter(predicate=pl.col("type") == f"eye{eye}|gazeorigin").select(pl.col("x", "y", "z")).to_numpy()[..., np.newaxis],
                dims=("time", "world_space", "eye"),
                coords=(
                    ("time", convert_timestamps(datatype=f"eye{eye}|gazeorigin")),
                    world_space_dim,
                    ("eye", np.array([eye], dtype=str)),
                ),
                attrs={
                    "info": "Position of the eye relative to the scene camera",
                },
            )
        )

        gaze_direction.append(
            xr.DataArray(
                data=data.filter(predicate=pl.col("type") == f"eye{eye}|gazedirection").select(pl.col("x", "y", "z")).to_numpy()[..., np.newaxis],
                dims=("time", "world_space", "eye"),
                coords=(
                    ("time", convert_timestamps(datatype=f"eye{eye}|gazedirection")),
                    world_space_dim,
                    ("eye", np.array([eye], dtype=str)),
                ),
                attrs={
                    "info": "The estimated gaze direction in 3D. The origin of this vector is the gazeorigin for the respective eye"
                },
            )
        )

        pupil_diameter.append(
            xr.DataArray(
                data=data.filter(predicate=pl.col("type") == f"eye{eye}|pupildiameter").select(pl.col("x")).to_numpy(),
                dims=("time", "eye"),
                coords=(
                    ("time", convert_timestamps(datatype=f"eye{eye}|pupildiameter")),
                    ("eye", np.array([eye], dtype=str)),
                ),
            )
        )

    gaze_data = xr.Dataset(
        data_vars={
            "gaze_video": gaze_video,
            "gaze_world": gaze_world,
            "gaze_origin": xr.concat(objs=gaze_origin, dim="eye"),
            "gaze_direction": xr.concat(objs=gaze_direction, dim="eye"),
            "pupil_diameter": xr.concat(objs=pupil_diameter, dim="eye"),
        },
    )

    # IMU data

    # the accelerometer and gyroscope seem to operate on the same clock, so represent those
    # in the same dataset and the magnetometer in its own array
    imu_accel_gyro_data = xr.Dataset(
        data_vars={
            imu_type: xr.DataArray(
                data=data.filter(predicate=pl.col("type") == imu_type).select(pl.col("x", "y", "z")).to_numpy(),
                dims=("time", "world_space"),
                coords=(
                    ("time", convert_timestamps(datatype=imu_type)),
                    world_space_dim,
                ),
                attrs=imu_attrs,
            )
            for (imu_type, imu_attrs) in {
                "accelerometer": {
                    "units": "m / s^2",
                    "info": "Acceleration in three dimensions in the coordinate system of the head unit",
                },
                "gyroscope": {
                    "units": "deg / s",
                    "info": "Rotation of the head unit",
                },
            }.items()
        },
    )

    imu_magnetometer_data = xr.DataArray(
        data=data.filter(predicate=pl.col("type") == "magnetometer").select(pl.col("x", "y", "z")).to_numpy(),
        dims=("time", "world_space"),
        coords=(
            ("time", convert_timestamps(datatype="magnetometer")),
            world_space_dim,
        ),
        attrs={
            "units": "microtesla",
            "info": "Magnetic field near the head unit",
        },
    )

    xr_data = {
        "gaze": gaze_data,
        "imu_accel_gyro": imu_accel_gyro_data,
        "imu_magnetometer": imu_magnetometer_data,
    }

    return xr_data

