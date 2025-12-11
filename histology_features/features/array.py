import re
import numpy
import os
from multiprocessing import RawArray
from joblib import Parallel, delayed
from tqdm import tqdm
import typing

def rescale_array(array, new_min, new_max):
    array = ((array - array.min()) / (array.max() - array.min())) * (new_max - new_min) + new_min
    return array

def extract_xy(filename):
    """Sort based on XY positions"""
    pattern = r'_\d+_\d+'
    match = re.findall(pattern, filename)[0]
    return int(match.split("_")[1]), int(match.split("_")[2])

def get_file_crop_idx(crop_directory):
    """Get the crop IDX from crops within a folder.
    Crop files are assumed to have the following nomenclature:
    _crop_x_y.ext
    
    For example, _crop_100_200.tiff"""


    files = os.listdir(crop_directory)

    output = numpy.zeros((len(files), 2)).astype(int)

    for i in range(len(files)):
        output[i] = extract_xy(files[i])

    return output

def numpy_to_raw_array(numpy_array):
    """Create a copy of a numpy array that is more efficiently
    usable by multiprocessing map. Converting to a raw array
    prevents the need for pickling a numpy array to each worker. 
    The RawArray created here can instead be read by all workers.
    
    The output array should not be modified directly."""
    
    # Create a RawArray for multiprocessing (ie. shared across workers) using the 
    # dtype of the original array
    raw_array = RawArray(numpy.ctypeslib.as_ctypes_type(numpy_array.dtype), numpy_array.size)
    
    # Consider the buffer as a numpy array
    raw_array_numpy = numpy.frombuffer(raw_array, dtype=numpy_array.dtype).reshape(numpy_array.shape)
    
    # Copy the data to the buffer/numpy array
    numpy.copyto(raw_array_numpy, numpy_array)

    return raw_array

def concatenate_npy_dir(npy_file_paths: str, load_fn: typing.Callable) -> numpy.ndarray:
    """Load and concatenate a series of npy files from a directory. Arrays 
    will be flattened before concatenation.

    Args:
        npy_file_paths (str): a list of .npy file paths to be concatenated
        load_fn (Callable): Function that will be used to load numpy arrays.
            Include any pre-processing, such as argmax here. 

    Returns:
        numpy.ndarray: Concatenated array
    """    
    # Load files in parallel
    arrays = Parallel(n_jobs=-1)(delayed(load_fn)(file) for file in tqdm(npy_file_paths))

    assert arrays[0].ndim == 1, "Ensure the load_fn flattens arrays."

    # Determine the total size of the output array
    total_shape = 0
    for arr in arrays:
        total_shape += arr.shape[0]

    # Pre-allocate an empty array
    result = numpy.empty((total_shape))

    # Assign arrays to regions of the pre-allocated array
    current_position = 0
    for arr in arrays:
        result[current_position:current_position + len(arr)] = arr
        current_position += len(arr)

    return result