"""
High-performance dataset handlers with memory caching capabilities.
Designed for scalable meteorological data processing.
"""

import logging
from abc import ABC, abstractmethod

# from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import xarray as xr


class DatasetHandler(ABC):
    """
    Abstract base class for high-performance dataset handlers.

    Provides unified interface for both on-demand fetching and memory caching
    of meteorological datasets with comprehensive error handling and monitoring.
    """

    def __init__(
        self,
        locations: List[str],
        data_shape: Tuple[int, ...] = (256, 256),
        base_paths: List[str] = [""],
        split: Optional[str] = None,
    ):
        self.base_paths = base_paths if base_paths else [""]
        self.locations = locations
        self.data_shape = data_shape
        self.split = split

        # Logger setup
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG)
        # Add file handler
        file_handler = logging.FileHandler("dataset_handler.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)

    @abstractmethod
    def fetch(self, datetimes: List[datetime], location: str, **kwargs) -> np.ndarray:
        """
        Fetch data from disk storage.

        Args:
            datetimes: List of datetime objects to fetch
            location: Geographic location identifier

        Returns:
            Numpy array containing the requested data
        """

    def _get_fallback_data(self, datetimes: List[datetime]) -> np.ndarray:
        """Return fallback data when fetch fails."""
        # Use data_shape if available, otherwise default shape
        print(f"Warning: getting fallback data from {self.__class__.__name__}")
        if len(datetimes) > 1:
            shape = (len(datetimes), *self.data_shape)
        else:
            shape = self.data_shape

        return np.zeros(shape, dtype=np.float32)

    def _find_valid_path(self, relative_path: str) -> Optional[Path]:
        """Find first existing path from base_paths."""
        for base_path in self.base_paths:
            full_path = Path(base_path) / relative_path
            if full_path.exists():
                return full_path
        return None


class GOES16Handler(DatasetHandler):
    """
    High-performance GOES-16 satellite data handler.

    Optimized for concurrent loading and memory-efficient caching
    of large satellite imagery datasets.
    """

    def __init__(
        self,
        locations: List[str],
        base_paths: List[str] = None,
        **kwargs,
    ):
        self.locations_dict = {
            "rio_de_janeiro": "SAT-ABI-L2-RRQPEF-rio_de_janeiro.hdf",
            "la_paz": "SAT-ABI-L2-RRQPEF-la_paz.hdf",
            "manaus": "SAT-ABI-L2-RRQPEF-manaus.hdf",
            "toronto": "SAT-ABI-L2-RRQPEF-toronto.hdf",
            "miami": "SAT-ABI-L2-RRQPEF-miami.hdf",
        }
        self.directory = "goes16_rrqpe"
        self.required_kwargs = []
        super().__init__(locations, (256, 256), base_paths)

    def fetch(self, datetimes: List[datetime], location: str, **kwargs) -> np.ndarray:
        """Fetch GOES-16 data from HDF5 files."""
        try:
            dataset_path = self._get_dataset_path(location)
            if not dataset_path:
                self.logger.warning(f"No GOES-16 data found for {location}")
                return self._get_fallback_data(datetimes)

            results = []
            with h5py.File(dataset_path, "r") as f:
                for dt in datetimes:
                    key = dt.strftime("%Y%m%d/%H%M")
                    if key in f:
                        data = f[key]
                        results.append(data)
                    else:
                        self.logger.debug(f"Missing GOES-16 data for {key}")
                        results.append(
                            np.zeros(self.data_shape, dtype=np.float32))

            return np.stack(results)

        except Exception as e:
            self.logger.error(f"GOES-16 fetch error: {e}")
            return self._get_fallback_data(datetimes)

    def _get_dataset_path(self, location: str) -> Optional[Path]:
        """Get path to GOES-16 dataset file."""
        filename = f"data/{self.directory}/{self.locations_dict[location]}"
        return self._find_valid_path(filename)


class IMERGHandler(DatasetHandler):
    """
    High-performance IMERG precipitation data handler.

    Handles NetCDF files with complex filename patterns and
    geographic coordinate transformations.
    """

    def __init__(
        self,
        locations: List[str],
        base_paths: List[str] = None,
        **kwargs,
    ):
        self.locations_dict = {
            "rio_de_janeiro": "final_cropped-lat=-35.7:-10.1,lon=-56.0:-30.4",
            "manaus": "final_cropped-lat=-15.9:9.7,lon=-72.8:-47.2",
            "la_paz": "final_cropped-lat=-29.3:-3.7,lon=-80.9:-55.3",
            "miami": "final_cropped-lat=13.0:38.6,lon=-93.0:-67.4",
            "toronto": "final_cropped-lat=30.9:56.5,lon=-92.2:-66.6",
        }
        self.required_kwargs = []
        super().__init__(locations, (256, 256), base_paths)

    def fetch(self, datetimes: List[datetime], location: str, **kwargs) -> np.ndarray:
        """Fetch IMERG data from NetCDF files."""
        try:
            results = []
            for dt in datetimes:
                filepath = self._get_imerg_filepath(dt, location)
                if filepath and filepath.exists():
                    data = self._read_imerg_file(filepath, location)
                    results.append(data)
                else:
                    self.logger.debug(f"Missing IMERG file for {dt}")
                    results.append(self.data_shape, dtype=np.float32)

            return np.stack(results)

        except Exception as e:
            self.logger.error(f"IMERG fetch error: {e}")
            return self._get_fallback_data(datetimes)

    def _get_imerg_filepath(self, dt: datetime, location: str) -> Optional[Path]:
        """Generate IMERG file path with complex naming convention."""
        shifted_dt = dt - timedelta(minutes=30)
        dt_minute = shifted_dt.minute // 30
        dir = f"data/imerg/{self.locations_dict[location]}/"
        file_name = (
            shifted_dt.strftime(
                "%Y%m%d") + f"-S{shifted_dt.hour:02d}{dt_minute * 30:02d}00-"
            f"E{shifted_dt.hour:02d}{dt_minute * 30 + 29:02d}59.nc"
        )
        return self._find_valid_path(dir + file_name)

    def _read_imerg_file(self, filepath: Path, location: str) -> np.ndarray:
        """Read and process IMERG NetCDF data."""
        try:
            with xr.open_dataset(filepath) as ds:
                return np.array(ds.precipitation)

        except Exception as e:
            self.logger.error(f"Error reading IMERG file {filepath}: {e}")
            return np.zeros(self.data_shape, dtype=np.float32)


class PredictionsHandler(DatasetHandler):
    """
    Handler for model prediction data (evonet, autoencoderklgan, earthformer).

    Handles HDF5 files containing predictions from various models with
    different key structures and data organizations.
    """

    def __init__(
        self,
        locations: List[str],
        dataset: str,
        base_paths: List[str] = None,
        split: Optional[str] = None,
    ):
        data_shape_map = {
            "evonet": (256, 256),
            "autoencoderklgan": (4, 32, 32),
            "earthformer": (4, 32, 32),
            "nowcastrio": (256, 256),
            "nowcastrio_autoenc": (8, 64, 64),
        }
        list_data = dataset.split("#")
        if len(list_data) in [3, 4]:
            self.model, self.original_data, self.hash_val = list_data[:3]
            self.ckpt_file = list_data[3] if len(list_data) == 4 else None
        else:
            raise ValueError(
                "Unexpected dataset format. Expected format: model#original_data#hash_val[#ckpt_file]")
        # Get model name
        model_name_map = {
            "evonet": "evolution_network",
            "autoencoderklgan": "autoencoderklgan",
            "earthformer": "earthformer",
            "nowcastrio": "nowcastrio",
            "nowcastrio_autoenc": "nowcastrio_autoenc",
        }
        data_shape = data_shape_map[self.model]
        self.model_name = model_name_map.get(self.model)
        self.required_kwargs = ["curr_dt", "split"]
        super().__init__(locations, data_shape, base_paths, split=split)

    def fetch(
        self, datetimes: List[datetime], location: str, curr_dt: Optional[datetime] = None, split: Optional[str] = None
    ) -> np.ndarray:
        """Fetch predictions data from HDF5 files."""
        if not split or not curr_dt:
            self.logger.error(
                "Missing required parameters for predictions fetch")
            return self._get_fallback_data(datetimes)

        try:
            # Construct file path
            if self.ckpt_file is not None:
                file_path = (
                    f"models/{self.model_name}/{self.hash_val}/predictions/"
                    f"{self.original_data}/{location}/predict_{self.ckpt_file}_{split}.hdf"
                )
            else:
                file_path = (
                    f"models/{self.model_name}/{self.hash_val}/predictions/"
                    f"{self.original_data}/{location}/predict_{split}.hdf"
                )

            # Find valid file path
            dataset_path = self._find_valid_path(file_path)
            if not dataset_path:
                self.logger.warning(f"Predictions file not found: {file_path}")
                return self._get_fallback_data(datetimes)

            results = []
            with h5py.File(dataset_path, "r") as hdf:
                if self.model == "earthformer" or self.model == "evonet" or self.model == "nowcastrio":
                    keys = [
                        f"{curr_dt.strftime('%Y-%m-%d %H:%M:%S')}/" f"{dt.strftime('%Y%m%d-%H%M')}" for dt in datetimes
                    ]
                    for key in keys:
                        if key in hdf:
                            pred = np.array(hdf[key])
                            results.append(pred)
                        else:
                            print("Getting fallback data for key:", key)
                            self.logger.debug(f"Key {key} not found")
                            results.append(
                                np.zeros(self.data_shape, dtype=np.float32))

                elif self.model == "autoencoderklgan":
                    keys = [
                        f"{dt.strftime('%Y-%m-%d %H:%M:%S')}" for dt in datetimes]
                    same_need_keys = [
                        f"{dt.strftime('%Y%m%d-%H%M')}" for dt in datetimes]
                    for key, n_key in zip(keys, same_need_keys):
                        try:
                            results.append(np.array(hdf[key][n_key]))
                        except KeyError:
                            print("Getting fallback data for key:", key, n_key)
                            self.logger.debug(f"Key {key}/{n_key} not found")
                            results.append(
                                np.zeros(self.data_shape, dtype=np.float32))
                elif self.model == "nowcastrio_autoenc":
                    keys = [
                        f"{dt.strftime('%Y-%m-%d %H:%M:%S')}" for dt in datetimes]
                    same_need_keys = [
                        f"{dt.strftime('%Y%m%d-%H%M')}" for dt in datetimes]
                    for key, n_key in zip(keys, same_need_keys):
                        try:
                            results.append(np.array(hdf[key][n_key]))

                        except KeyError:
                            print("Getting fallback data for key:", key, n_key)
                            self.logger.debug(f"Key {key}/{n_key} not found")
                            results.append(
                                np.zeros(self.data_shape, dtype=np.float32))

            return np.stack(results)

        except Exception as e:
            self.logger.error(f"Predictions fetch error: {e}")
            return self._get_fallback_data(datetimes)


class FieldsIntensitiesHandler(DatasetHandler):
    """
    Handler for fields intensities data combining motion fields and intensities.

    Processes HDF5 files containing motion fields and intensity data
    concatenated along the first axis for meteorological analysis.
    """

    def __init__(
        self,
        locations: List[str],
        dataset: str,
        base_paths: List[str] = None,
        **kwargs,
    ):
        dataset_split = dataset.split("#")
        self.dataset_base = dataset_split[0]
        assert self.dataset_base == "fields_intensities", "Dataset must start with 'fields_intensities'"
        self.dataset = dataset_split[1]
        self.method = dataset_split[2]
        super().__init__(locations, (3, 256, 256), base_paths)

    def fetch(self, datetimes: List[datetime], location: str, **kwargs) -> np.ndarray:
        """Fetch fields intensities data from HDF5 files."""
        try:
            file_path = f"data/{self.dataset_base}/{self.method}/{self.dataset}_{location}.hdf"
            dataset_path = self._find_valid_path(file_path)

            if not dataset_path:
                self.logger.warning(
                    f"Fields intensities file not found: {file_path}")
                return self._get_fallback_data(datetimes)

            results = []
            with h5py.File(dataset_path, "r") as hdf:
                keys = [dt.strftime("%Y%m%d/%H%M") for dt in datetimes]

                for key in keys:
                    try:
                        # Get intensities and add new axis
                        intensities = np.squeeze(hdf["intensities"][key])
                        intensities = intensities[np.newaxis, :, :]

                        # Get motion fields
                        fields = hdf["motion_fields"][key]

                        # Concatenate along first axis
                        full_tensor = np.concatenate(
                            (fields, intensities), axis=0)
                        results.append(full_tensor)

                    except KeyError:
                        self.logger.debug(
                            f"Key {key} not found for {location}")
                        # Create fallback: 2 motion fields + 1 intensity field
                        arr = np.zeros(self.data_shape, dtype=np.float32)
                        results.append(arr)

            return np.stack(results)

        except Exception as e:
            self.logger.error(f"Fields intensities fetch error: {e}")
            return self._get_fallback_data(datetimes)


class DatasetHandlerFactory:
    """
    Factory class for creating appropriate data handlers based on dataset type.
    Supports all data sources from the original fetch_dataset_tensor module.
    """

    _handlers = {
        "goes16_rrqpe": GOES16Handler,
        "imerg": IMERGHandler,
        "evonet": PredictionsHandler,
        "autoencoderklgan": PredictionsHandler,
        "earthformer": PredictionsHandler,
        "fields_intensities": FieldsIntensitiesHandler,
        "nowcastrio_autoenc": PredictionsHandler,
        "nowcastrio": PredictionsHandler,
    }

    @classmethod
    def create_handler(
        cls,
        dataset: str,
        locations: List[str],
        base_paths: List[str] = None,
        split: Optional[str] = None,
    ) -> DatasetHandler:
        """Create appropriate handler for dataset type."""

        # Handle model predictions (evonet, autoencoderklgan, earthformer)
        dataset_split = dataset.split("#")
        dataset_base = dataset_split[0]
        if len(dataset_split) > 1:
            return cls._handlers[dataset_base](locations, dataset, base_paths, split=split)

        # Handle standard datasets
        try:
            handler_class = cls._handlers[dataset]
            return handler_class(locations, base_paths)
        except KeyError:
            raise ValueError(f"Unknown dataset: {dataset}")

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""
        return list(cls._handlers.keys())
