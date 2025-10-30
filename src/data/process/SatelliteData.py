from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate


class SatelliteData:
    def __init__(
        self,
        data: pd.DataFrame,
        product: list[str],
        value: str | None = None,
    ) -> None:
        self.data = data
        self.product = product
        self.value = value

    @classmethod
    def load_data(cls, input_filepath: Path | str, product: str | None = None, value: str | None = None):
        return cls(
            pd.read_feather(input_filepath),
            product,
            value,
        )

    def cast_to_float32(self):
        for col in self.data.columns:
            if self.data[col].dtype == "float64":
                self.data[col] = self.data[col].astype(np.float32)

    def interp_at_grid(self, band: str, timestamp: datetime, target_grid: NDArray):
        assert (timestamp >= self.data["creation"]).any(
        ), "Timestamp passed precedes all timestamps in the data"
        closest_timestamp = self.data.loc[self.data["creation"]
                                          <= timestamp, "creation"].max()
        df = self.data[self.data["creation"] == closest_timestamp]
        column = f"{self.value}_{band}" if self.product == "ABI-L2-MCMIPF" else self.value

        x = np.array(df["lon"])
        y = np.array(df["lat"])
        values = np.array(df[column])
        nan_indices = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~nan_indices]
        y = y[~nan_indices]
        values = values[~nan_indices]
        points = np.stack((x, y)).T
        shape = target_grid.shape[:2]
        try:
            interp_values = interpolate.griddata(
                points,
                values,
                (target_grid[:, :, 1].flatten(),
                 target_grid[:, :, 0].flatten()),
                method="linear",
            ).reshape(shape)
        except ValueError:
            return np.zeros(shape)

        return interp_values
