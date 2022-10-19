"""
Parsing utils. Read json, command-line and external files defining parsed
parameters.
"""
from abc import ABC, abstractmethod
import argparse
import importlib
import h5py
import glob
import json
import os
import pandas
import re
import torch
import shutil
import PIL.Image as Image

from typing import Dict, Iterable, List, Optional, TypeVar, Union, Tuple, Any

Array = Iterable[Iterable[float]]
PandasDataFrame = TypeVar('PandasDataFrame')
Model = TypeVar('Model')


class IOPipe(ABC):
    """Abstract class designed to read and write from external sources.
    This class does not store any instance of the actual class."""
    @abstractmethod
    def read(self, *args, **kwargs) -> Union[Dict, PandasDataFrame, Array]:
        """Read from a source and load into working memory."""

    @abstractmethod
    def write(self, to_file: str, data: Any, *args, **kwargs):
        """Write a formatted source of data to an external file."""


class CSVPipe(IOPipe):
    """Provides functionality to read and write CSV/TXT files."""
    def read(self, file: str, **kwargs) -> PandasDataFrame:
        return pandas.read_csv(file, **kwargs)

    def write(self, to_file: str, data: PandasDataFrame, **kwargs) -> None:
        data.to_csv(to_file, **kwargs)


class H5Pipe(IOPipe):
    """HDF5 format through the h5py library.
    TODO (ricardokleinlein): Automatically detect all fields in h5file."""
    def read(self, file: str, key: Optional[Union[int, str]] = None,
             **kwargs) -> Tuple[Any, ...]:
        """
        TODO
        Args:
            file:
            key:
            **kwargs:

        Returns:

        """
        h5file = h5py.File(file, mode="r")
        if key is not None:
            if not isinstance(key, list):
                key = [key]
        out = tuple([h5file[k][:] for k in key])
        h5file.close()
        return out

    def write(self, to_file: str, data: Array, tag: str, **kwargs) -> None:
        """
        TODO
        Args:
            to_file:
            data:
            tag:
            **kwargs:

        Returns:

        """
        h5file = h5py.File(to_file, 'r+')
        self.save_h5(h5file, data, tag)
        h5file.close

    def save_h5(hdf5: h5py.File, data: Array, name: str) -> h5py.File:
        """Add data to an existing H5File object.

        Args:
            hdf5: Object to write data to.
            data: Data to write.
            name: Label under which data is saved.

        Returns:
            Updated h5py.File object.
        """
        try:
            hdf5.create_dataset(name, data=data, chunks=True,
                                compression='gzip')
        except TypeError:
            data_type = h5py.special_dtype(vlen=str)
            hdf5.create_dataset(name, data=data, chunks=True,
                                compression='gzip', dtype=data_type)
        return hdf5


class Mp4Pipe(IOPipe):
    """I/O Interface for videos in mp4 format."""
    def read(self, file: str, fps: int = None, keep_temp: bool = False,
             **kwargs) -> Array:
        """Extract the frames of a video file at a certain rate.

        NOTE: In order to work, you must have a correct installation of
        ffmpeg in your system.

        Args:
            file: Path to video file.
            fps: Frames Per Second.
            keep_temp: Whether to delete from disk the extracted frames

        Returns:
            PIL.Images extracted from the video.
        """
        regex_videos = re.compile(r'videos', flags=re.IGNORECASE)
        temp_dir = re.sub(regex_videos, 'frames', file).replace('.mp4', '')
        os.makedirs(temp_dir, exist_ok=True)
        cmd = f'ffmpeg -i {file}'
        cmd += f' -vf "fps={fps}" ' if fps else ' '
        cmd += f'{temp_dir}/%05d.jpeg -loglevel quiet'
        os.system(cmd)
        images = [Image.open(item) for item in glob.glob(temp_dir + '/*.jpeg')]
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return images

    def write(self, to_file: str, **kwargs) -> None:
        raise NotImplementedError    # TODO (ricardokleinlein)


class CmdLinePipe(IOPipe):
    """Provides functionality to parse and export command-line-typed
    options."""
    parser = argparse.ArgumentParser(
        description='Command-line Parser for DL instructions',
        formatter_class=argparse.RawTextHelpFormatter
    )

    def read(self) -> Tuple[Dict, Dict]:
        self.parser.add_argument("config",
                                 type=str,
                                 help="Path to configuration file")
        config, unknown = self.parser.parse_known_args()
        unk_formatted = self._sort_flag_options(unknown)
        return config.__dict__['config'], unk_formatted

    def write(self, to_file: str, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _sort_flag_options(options: List) -> Dict:
        """Extra options are read as a list. Turn them into a sorted dict."""
        keys = [item.lstrip('--') for item in options if item.startswith('--')]
        vals = [item for item in options if not item.startswith('--')]
        return {k: v for k, v in zip(keys, vals)}


class JsonPipe(IOPipe):
    """Provides functionality to work with json files."""

    def read(self, file: str, **kwargs) -> Dict:
        with open(file, 'r') as jfile:
            data = json.load(jfile)
        return data

    def write(self, to_file: str, data: dict) -> None:
        with open(to_file, 'w') as jfile:
            json.dump(data, jfile)


class ConfigSolver:
    """Solve manual overwriting of json fields in the command-line.

    Attributes:
        root: Package base level path.

    TODO: Add new manually-entered parameters via command line.
    """
    json_reader = JsonPipe()
    cmd_reader = CmdLinePipe()

    def __init__(self, base_path: str) -> None:
        """Initialization based on base directory of the package."""
        self.root = base_path

    def __call__(self) -> Dict:
        """Read a json file, and overrides any field by its cmd-line
        counterpart.

        Returns:
            Configuration parameters of a DL experiment.
        """
        config, cmd_data = self.cmd_reader.read()
        json_data = self.json_reader.read(os.path.join(self.root, config))
        overwritten = json_data.keys() & cmd_data.keys()
        for field in overwritten:
            json_data[field] = cmd_data[field]
        return json_data


class TorchDatasetPipe(IOPipe):
    """I/O operations over Pytorch map-style custom_datasets."""
    def read(self, root: str, module: str, input_transform: str = None,
             label_transform: str = None):
        """TODO

        Args:
            root:
            module:
            input_transform:
            label_transform:

        Returns:

        """
        mod, name = module.split('.')
        dataset_classname = getattr(
            importlib.import_module('custom_datasets.' + mod), name)
        dataset = dataset_classname(root)
        # TODO: Worth it? reconsider
        # if input_transform is not None:
        #     dataset.set_transform(getattr(transforms, input_transform))
        # if label_transform is not None:
        #     dataset.set_label_transform(label_transform)
        return dataset

    def write(self):
        raise NotImplementedError


class TorchModelPipe(IOPipe):
    """I/O operations for Pytorch nn.Module-based models."""
    def read(self, module: str, conf: Dict) -> Model:
        mod, arch = module.split('.')
        arch_class = getattr(importlib.import_module('nn.' + mod), arch)
        model_instance = arch_class(**conf)
        return model_instance

    def write(self, to_file: str, data: Dict) -> None:
        if os.path.isfile(to_file):
            os.remove(to_file)
        torch.save(data, to_file)
