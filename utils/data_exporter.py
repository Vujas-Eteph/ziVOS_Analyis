'''
Support different extension formats for saving numpy arrays.

by Stéphane Vujasinovic
'''
# - IMPORTS ---
import numpy as np
import pandas as pd
import h5py
import zarr
import json


# - CLASS ---
class DataExporter():
    '''
    Used to save numpy arrays ONLY (atm).
    Currently supported formats are:
        - HDF5: 'HDF5'
    '''

    _supported_extensions = ["HDF5", "NPZ", "NPY", "Parquet", "Zarr",
                             "Feather"]

    def __init__(
        self
    ):
        """Initialization"""
        self._file_extension = "HDF5"

    @property
    def file_extension(
        self
    ):
        """Getter method"""
        # Getter method
        return self._file_extension

    @file_extension.setter
    def file_extension(
        self,
        value: str
    ):
        """Setter method"""
        if value not in DataExporter._supported_extensions:
            print(f"Warning: '{value}' is not a supported file extension."
                  f"Reverting to default 'HDF5'.")
            self._file_extension = 'HDF5'
        else:
            self._file_extension = value

    def save_data(
        self,
        data_to_save: np.ndarray,
        name: str,
        filename: str,
        file_extension=None
    ):
        if file_extension is None:
            file_extension = self._file_extension
        """Carrefour function for saving data."""
        if "HDF5" == file_extension:
            filename_ext = f"{filename}.h5"
            self.save_to_HDF5_format(data_to_save, name, filename_ext)
        elif "NPZ" == file_extension:
            filename_ext = f"{filename}.npz"
            self.save_to_NPZ_format(data_to_save, filename_ext)
        elif "NPY" == file_extension:
            filename_ext = f"{filename}.npy"
            self.save_to_NPY_format(data_to_save, filename_ext)
        elif "Parquet" == file_extension:
            raise ('Not functioning atm')
            filename_ext = f"{filename}.parquet"
            df = pd.DataFrame(data_to_save)
            self.save_to_Parquet_format(df, filename_ext)
        elif "Zarr" == file_extension:
            filename_ext = f"{filename}.zarr"
            self.save_to_Zarr_format(data_to_save, filename_ext)
        elif "Feather" == file_extension:
            raise ('Not functioning atm')
            filename_ext = f"{filename}.feather"
            df = pd.DataFrame(data_to_save)
            self.save_to_Feather_format(df, filename_ext)
        else:
            raise ValueError("Could not save the data")
        

    def read_data(
        self,
        name: str,
        filename: str
    ) -> np.ndarray:
        """Carrefour function for reading data."""
        if 'HDF5' == self._file_extension:
            filename_ext = f"{filename}.h5"
            return self.read_from_HDF5_format(name, filename_ext)
        elif "NPZ" == self._file_extension:
            filename_ext = f"{filename}.npz"
            return self.read_from_NPZ_format(filename_ext)
        elif "NPY" == self._file_extension:
            filename_ext = f"{filename}.npy"
            return self.read_from_NPY_format(filename_ext)
        elif "Parquet" == self._file_extension:
            filename_ext = f"{filename}.parquet"
            df = self.read_from_Parquet_format(filename_ext)
            return df.to_numpy()
        elif "Zarr" == self._file_extension:
            filename_ext = f"{filename}.zarr"
            return self.read_from_Zarr_format(filename_ext)
        elif "Feather" == self._file_extension:
            filename_ext = f"{filename}.feather"
            df = self.read_from_Feather_format(filename_ext)
            return df.to_numpy()
        else:
            raise ValueError("Could not read the data")

    # - 1] Functions for HDF5 (Hierarchical Data Format version 5) ---
    @staticmethod
    def save_to_HDF5_format(
        data_to_save: np.ndarray,
        name: str,
        filename: str
    ):
        """Save NumPy array in HDF5 file, compressiong ratio by default = 4."""
        with h5py.File(filename, 'w') as hdf:
            hdf.create_dataset(name,
                               data=data_to_save,
                               compression='gzip')

    @staticmethod
    def read_from_HDF5_format(
        name: str,
        filename: str,
    ) -> np.ndarray:
        """Read the data in a HDF5 file."""
        with h5py.File(filename, 'r') as hdf:
            data = hdf[name][:]
            return data

    # - 2] Functions for NPZ (NumPy Zipped File) ---
    @staticmethod
    def save_to_NPZ_format(
        data_to_save: np.ndarray,
        filename: str
    ):
        """Save NumPy array in NPZ (compressed) file format."""
        np.savez_compressed(filename, data_to_save)

    @staticmethod
    def read_from_NPZ_format(
        filename: str
    ) -> np.ndarray:
        """Read data from an NPZ file."""
        with np.load(filename) as data:
            return next(iter(data.values()))

    # - 3] Functions for NPY (NumPy Binary File) ---
    @staticmethod
    def save_to_NPY_format(
        data_to_save: np.ndarray,
        filename: str
    ):
        """Save NumPy array in NPY file format."""
        np.save(filename, data_to_save)

    @staticmethod
    def read_from_NPY_format(
        filename: str
    ) -> np.ndarray:
        """Read data from an NPY file."""
        return np.load(filename)

    # - 4] Functions for Parquet ---
    @staticmethod
    def save_to_Parquet_format(
        df_to_save: pd.DataFrame,
        filename: str
    ):
        """Save DataFrame in Parquet file format."""
        df_to_save.to_parquet(filename)

    @staticmethod
    def read_from_Parquet_format(
        filename: str
    ) -> pd.DataFrame:
        """Read data from a Parquet file."""
        return pd.read_parquet(filename)

    # - 5] Functions for Zarr ---
    @staticmethod
    def save_to_Zarr_format(
        data_to_save: np.ndarray,
        filename: str
    ):
        """Save NumPy array in Zarr file format."""
        zarr.save(filename, data_to_save)

    @staticmethod
    def read_from_Zarr_format(
        filename: str
    ) -> np.ndarray:
        """Read data from a Zarr file."""
        return zarr.load(filename)

    # - 6] Functions for Feather ---
    @staticmethod
    def save_to_Feather_format(
        df_to_save: pd.DataFrame,
        filename: str
    ):
        """Save DataFrame in Feather file format."""
        df_to_save.to_feather(filename)

    @staticmethod
    def read_from_Feather_format(
        filename: str
    ) -> pd.DataFrame:
        """Read data from a Feather file."""
        return pd.read_feather(filename)

    # - 6] Functions for JSON ---
    @staticmethod
    def save_to_JSON_format(
        data: dict,
        filename: str
    ):
        """Save a dict to a JSON file format."""
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def read_from_JSON_format(
        filename: str
    ) -> dict:
        """Read data from a JSON file and return it as a dictionary."""
        with open(filename, 'r') as file:
            return json.load(file)
