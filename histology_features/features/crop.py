import functools
import os
import pathlib
import re
import shutil
import typing
from multiprocessing import Pool, cpu_count

import dask
import geopandas
import numpy
import pandas
import polars
import rasterio
import skimage
from rasterio.features import rasterize
from sklearn.model_selection import train_test_split

from histology_features.features.array import numpy_to_raw_array
from histology_features.features.image import load_tiff_scale, load_zarr_scale
from histology_features.features.transcripts import (
    scale_and_round_xenium_transcripts,
)
from histology_features.polygons.polygons import (
    create_geopandas_crops,
    scale_spatialdata_polygons_and_add_cluster_id,
)
from histology_features.spec.xenium import Xenium
from histology_features.utility.colormap import load_gray2rgb_save
from histology_features.utility.utils import remove_array_overlap


def array_crop(
    array,
    top: int,
    left: int,
    crop_window_size: tuple,
    padding_value: int = 0,
    dimensions: typing.Dict[str, int] = {"c": -1, "y": 0, "x": 1},
):
    """Crop a dask array. If a crop of the requested size at the
    requested location is not available, the image will be padded.

    Args:
        array (_type_): Dask array
        top (int): Top of the bounding box (ie. the top y-intercept)
        left (int): Left of the bounding box (ie. the left x-intercept)
        crop_window_size (tuple): Size of the crop window
        padding_value (int, optional): Value to pad with. Defaults to 0.
        dimensions: Dictionary assigning dimensions CYX to array indices. Defaults to {"c": -1, "y": 0, "x": 1}.

    Returns:
        _type_: Dask array
    """
    height, width = array.shape[dimensions["y"]], array.shape[dimensions["x"]]

    bottom = top + crop_window_size[dimensions["y"]]
    right = left + crop_window_size[dimensions["x"]]

    slices = [slice(None)] * array.ndim
    if array.ndim == 3:
        slices[dimensions["c"]] = slice(None, None, None)
    slices[dimensions["y"]] = slice(max(top, 0), bottom, None)
    slices[dimensions["x"]] = slice(max(left, 0), right, None)
    slices = tuple(slices)

    # Pad the image if the crop box is negative
    if left < 0 or top < 0 or right > width or bottom > height:
        if array.ndim == 3:
            padding = numpy.zeros((3, 2))
            # Add 0 padding to the C dimension
            padding[dimensions["c"]] = [0, 0]
            padding[dimensions["y"]] = [
                max(-top + min(0, bottom), 0),
                max(bottom - max(height, top), 0),
            ]
            padding[dimensions["x"]] = [
                max(-left + min(0, right), 0),
                max(right - max(width, left), 0),
            ]
        else:
            padding = numpy.zeros((2, 2))
            padding[dimensions["y"]] = [
                max(-top + min(0, bottom), 0),
                max(bottom - max(height, top), 0),
            ]
            padding[dimensions["x"]] = [
                max(-left + min(0, right), 0),
                max(right - max(width, left), 0),
            ]

        # Cast to int
        padding = padding.astype(int)
        # For some reason, Dask pad doesn't parse padding as a numpy array
        # So, convert it to a tuple of tuples
        padding = tuple(map(tuple, padding))
        # Pad the array with the required padding
        array = numpy.pad(
            array=array,
            pad_width=padding,
            mode="constant",
            constant_values=padding_value,
        )
    return array[slices]


def skip_dask_blocks(
    mask_block,
) -> bool:
    """Return True if a dask block contains foreground, which is a
    non-zero integer.

    Args:
        mask_block (_type_): Dask block

    Returns:
        bool: True if foreground, False if background
    """
    if isinstance(mask_block, dask.array.Array):
        if mask_block.max().compute() == 0:
            return False
        else:
            return True
    # Just in case it's a numpy array
    else:
        if mask_block.max() == 0:
            return False
        else:
            return True


def get_crop_indices(
    image_shape: typing.Tuple[int],
    crop_window_size: typing.Tuple[int, int],
    dimensions: typing.Dict[str, int] = {"c": -1, "y": 0, "x": 1},
    overlap: float = 0.0,
) -> numpy.typing.NDArray:
    """Find the indices for the top and left of crops of a defined
    size for a particular image, allowing for overlap.

    Args:
        image_shape (typing.Tuple[int]): Image shape.
        crop_window_size (typing.Tuple[int, int]): Size of the cropping window in YX format.
        dimensions (dict, optional): Dictionary assigning dimensions CYX to array indices.
            Defaults to {"c": -1, "y": 0, "x": 1}.
        overlap (float, optional): Amount of overlap between crops as a fraction of crop_window_size.
            Should be between 0 and 1. Defaults to 0.0 (no overlap).

    Returns:
        numpy.typing.NDArray: All possible top and left crop indices for a given image shape.
    """
    img_height, img_width = (
        image_shape[dimensions["y"]],
        image_shape[dimensions["x"]],
    )

    # Adjust step size based on overlap
    step_height = int(crop_window_size[0] * (1 - overlap))
    step_width = int(crop_window_size[1] * (1 - overlap))

    # Ensure steps are at least 1 pixel
    step_height = max(1, step_height)
    step_width = max(1, step_width)

    # Compute crop positions with adjusted steps
    y_positions = numpy.arange(0, img_height - crop_window_size[0] + 1, step_height)
    x_positions = numpy.arange(0, img_width - crop_window_size[1] + 1, step_width)

    # Meshgrid to get all combinations of y and x positions
    all_crops = numpy.array(numpy.meshgrid(y_positions, x_positions)).T.reshape(-1, 2)

    return all_crops


def crop_array_and_save(
    array: numpy.ndarray,
    top_and_left_idx: typing.Tuple[int, int],
    crop_window_size: typing.Tuple[int, int],
    padding_value: int = 0,
    dimensions: typing.Dict[str, int] = {"c": -1, "y": 0, "x": 1},
    filename_prefix: str = None,
    save_dir: str = None,
) -> None:
    """Crop an array and save.

    Args:
        array (numpy.ndarray): Array to be cropped
        top_and_left_idx (typing.Tuple[int, int]): Top and left array indices for the crop window.
        crop_window_size (typing.Tuple[int, int]): Crop window size in YX.
        padding_value (int, optional): Value to pad the image with if cropping window spans
        outside the image dimensions. Defaults to 0.
        dimensions (_type_, optional): Dictionary assigning dimensions CYX to array indices. Defaults to {"c": -1, "y": 0, "x": 1}.
        filename_prefix (str, optional): Prefix to add to the crop name. Defaults to None.
        save_dir (str, optional): Path to save the crop. Defaults to None.
    """
    crop = array_crop(
        array=array,
        top=top_and_left_idx[0],
        left=top_and_left_idx[1],
        crop_window_size=crop_window_size,
        padding_value=padding_value,
        dimensions=dimensions,
    )
    skimage.io.imsave(
        os.path.join(
            save_dir,
            f"{filename_prefix or ''}_crop_{top_and_left_idx[0]}_{top_and_left_idx[1]}.tiff",
        ),
        crop,
        check_contrast=False,
    )


def crop_dataframe_and_save(
    df,
    top_and_left_idx: typing.Tuple[int, int],
    crop_window_size: typing.Tuple[int, int],
    transcript_downsample_factor: int = 0,
    padding_value: int = 0,  # Not used since it'll be 0 anyway
    dimensions: typing.Dict[str, int] = {"c": -1, "y": 0, "x": 1},
    filename_prefix: str = None,
    save_dir: str = None,
    x_column_name: str = "x_location",
    y_column_name: str = "y_location",
    z_column_name: str = "feature_name",
    z_idx_column_name: str = "feature_name_idx",
    pixel_value_column_name: str = "pixel_value",
    z_dimension_length: int = None,
) -> None:
    """
    transcript_downsample_factor: Integer value to downsample to.
        This represents the scale_level the transcripts should be scaled to.
        The value will be transformed to 2**transcript_downsample_factor
    """

    if z_dimension_length is None:
        z_dimension_length = len(df[z_column_name].unique())

    crop_window_size = numpy.array(crop_window_size)

    crop = dataframe_crop(
        df=df,
        # Here we divide by the downsample factor to create
        # smaller (ie. denser) crops of an already downsampled
        # transcriptomics dataframe. This results in smaller crops
        # for transcriptomics if transcript_downsample_factor > 1.
        # We also keep top_and_left_idx so we can assign it the same
        # filename as non-downsampled modalities.
        top=numpy.round(
            top_and_left_idx[0] / 2**transcript_downsample_factor, 0
        ).astype(int),
        left=numpy.round(
            top_and_left_idx[1] / 2**transcript_downsample_factor, 0
        ).astype(int),
        # top=top_and_left_idx[0],
        # left=top_and_left_idx[1],
        x_column_name=x_column_name,
        y_column_name=y_column_name,
        z_idx_column_name=z_idx_column_name,
        pixel_value_column_name=pixel_value_column_name,
        z_dimension_length=z_dimension_length,
        crop_window_size=numpy.round(
            crop_window_size / 2**transcript_downsample_factor, 0
        ).astype(int),
        # crop_window_size=crop_window_size,
        dimensions=dimensions,
    )

    numpy.save(
        os.path.join(
            save_dir,
            f"{filename_prefix or ''}_crop_{top_and_left_idx[0]}_{top_and_left_idx[1]}.npy",
        ),
        crop,
        allow_pickle=True,
    )


def rasterize_geometries_and_save(
    geo_dataframe_crop: typing.Dict[
        typing.Tuple[int, int], geopandas.GeoDataFrame
    ],
    crop_window_size: typing.Tuple[int, int],
    save_dir: str,
    filename_prefix: str = None,
) -> None:
    """Crop and rasterize polygon geometries into a numpy array
    and save.

    Args:
        geo_dataframe (geopandas.GeoDataFrame): GeoPandas DataFrame containing the polygons to be rasterized
        top_and_left_idx (typing.Tuple[int, int]): Top and left cropping indices
        crop_window_size (typing.Tuple[int, int]): Crop window size
        save_dir (str): Directory to save the crop
        filename_prefix (str, optional): Optional crop name to prepend the top and left indices. Defaults to None.
    """

    assert len(geo_dataframe_crop) == 1, "Expected a dictionary of length 1"

    top_and_left_idx = list(geo_dataframe_crop.keys())[0]
    geo_dataframe = geo_dataframe_crop[top_and_left_idx]

    # List copy
    shapes = [
        (geom, value)
        for geom, value in zip(
            geo_dataframe["geometry"], geo_dataframe["cell_type"]
        )
    ]

    # Define the origin on which to transform the polygons
    # ie. translate the polygon coords to the crop area
    top, left = top_and_left_idx
    bottom, right = top + crop_window_size[0], left + crop_window_size[1]
    transform = rasterio.transform.from_origin(left, bottom, 1, 1)

    # Rasterize the cropped polygons
    crop = rasterize(
        shapes=shapes,
        out_shape=crop_window_size,
        transform=transform,
        fill=0,
        dtype="int32",
    )[
        ::-1, :
    ]  # Inverse the y-axis

    skimage.io.imsave(
        os.path.join(
            save_dir,
            f"{filename_prefix or ''}_crop_{top_and_left_idx[0]}_{top_and_left_idx[1]}.tiff",
        ),
        crop,
        check_contrast=False,
    )


def save_masked_crops(
    image,
    crop_window_size,
    save_dir=None,
    mask=None,
    dimensions: dict = {"c": -1, "y": 0, "x": 1},
    filename_prefix: str = None,
    crop_scaling_factor: float = None,
    order: int = 0,
):
    if image.ndim == 3:
        assert (
            image.shape[dimensions["c"]] == 3
        ), "Expected the channel dimension to be in the 0th position."

    crop_idx = get_crop_indices(
        image_shape=image.shape,
        crop_window_size=crop_window_size,
        dimensions=dimensions,
    )

    cropped_images = {
        f"{idx[0]}_{idx[1]}": array_crop(
            image,
            left=idx[1],
            top=idx[0],
            crop_window_size=crop_window_size,
            dimensions=dimensions,
        )
        for idx in crop_idx
    }
    if mask is not None:
        cropped_masks = {
            f"{idx[0]}_{idx[1]}": array_crop(
                mask,
                left=idx[1],
                top=idx[0],
                crop_window_size=crop_window_size,
                dimensions=dimensions,
            )
            for idx in crop_idx
        }
        # Remove background crops
        cropped_images = {
            k: v
            for (k, v), (_, mask_v) in zip(
                cropped_images.items(), cropped_masks.items()
            )
            if skip_dask_blocks(mask_v)
        }

    if save_dir is None:
        save_dir = "."
    else:
        os.makedirs(save_dir, exist_ok=True)

    for i, (k, crop) in enumerate(cropped_images.items()):
        if crop_scaling_factor is not None:
            # If we are rescaling, we rescale the array **and** the crop indices.
            # This allows us to later align regions between different scales
            # intuitively, since multiple scales will have the same filename crop idx.

            # Update the key with the equivalent crop indices of the resized crop
            scaled_crop_idx = (crop_idx / crop_scaling_factor).astype(int)[i]
            k = f"{scaled_crop_idx[0]}_{scaled_crop_idx[1]}"
            # Rescale the crop based on the scaling factor
            # Also, inverse the scaling since we're now scaling UP
            crop = skimage.transform.rescale(
                crop,
                scale=crop_scaling_factor**-1,
                channel_axis=-1,
                order=order,
            )
        skimage.io.imsave(
            os.path.join(save_dir, f"{filename_prefix or ''}_crop_{k}.tiff"),
            crop,
            check_contrast=False,
        )


def train_test_split_crops(
    modality_crop_paths: typing.Dict[str, typing.List[str]],
    train_val_test_sizes: typing.Tuple[float, float, float] = (
        0.85,
        0.10,
        0.05,
    ),
    unique_filename_regex: str = r"_\d+_\d+",
    random_state=42,
    add_subfolder: bool = True,
):
    """For a series of multimodal crops, split them into train, test and validation
    sets.

    Train and validation are typically used during model training and the test set is
    withheld for later evaluation. The validation set is not used for training.


    Args:
        modality_crop_paths (typing.Dict[str, typing.List[str]]): Dictionary containing the parent paths for each
        modality that is to be split. Files inside this directory will be assumed to be crops.
        All paths must be present for all modalities. A check for this will occur to ensure there is nothing missing.
        train_val_test_sizes (typing.Tuple[float, float, float], optional): Split proprotion. Defaults to (0.85, 0.10, 0.05).
        unique_filename_regex (str, optional): Regex that returns a string that is unique for each crop.
        Default assumes filenames with style: _crop_x_y.fileextension. Defaults to r"_\d+_\d+".
        random_state (int, optional): Random seem for split. Defaults to 42.
        add_subfolder (bool, optional): Optionally add the subdirectories that a model expects. Defaults to True.

    Raises:
        ValueError: Missing crop.
    """
    # Sizes of datasets
    train_size, val_size, test_size = train_val_test_sizes

    # Find the paths for crops
    modality_file_lists = {}
    for modality_key, pth in modality_crop_paths.items():
        modality_file_lists[modality_key] = [
            os.path.join(pth, file) for file in os.listdir(pth)
        ]

    modality_file_names = {}
    # Check that matching files are present in each directory
    for modality_key, file_list in modality_file_lists.items():
        # Make sure we just extract the file names and not risk
        # the regex picking up something in the full path
        file_list = [pathlib.Path(i).name for i in file_list]

        # For each modality, create a list of all the file_names
        modality_file_names[modality_key] = re.findall(
            unique_filename_regex, "".join(file_list)
        )
        # Sort the list
        modality_file_names[modality_key] = sorted(
            modality_file_names[modality_key]
        )

    first_modality_name = list(modality_file_names.keys())[0]

    # Validate that all crops are present
    assert all(
        i == modality_file_names[first_modality_name]
        for i in modality_file_names.values()
    ), "Missing crops in the provided folders."

    # At this stage, we know that all crops are the same for each modality.
    # We can then decide which crops to train/val/test with.
    # Here, we split the unique filenames, rather than the paths themselves
    train_crops, val_crops = train_test_split(
        modality_file_names[first_modality_name],
        train_size=train_size + test_size,
        random_state=random_state,
    )
    train_crops, test_crops = train_test_split(
        train_crops, train_size=train_size + val_size, random_state=random_state
    )

    print(
        f"Number of train crops (per modality): {len(train_crops)}\nNumber of val crops (per modality): {len(val_crops)}\nNumber of test crops (per modality): {len(test_crops)}"
    )

    for modality_key, file_paths in modality_file_lists.items():
        for file in file_paths:
            # Only consider the actual filename
            file_name = pathlib.Path(file).name
            # Get the crop id, which is the crop indices
            crop_id = re.findall(unique_filename_regex, file_name)[0]
            # Get parent path that's one up from direct parent
            parent_dir = str(pathlib.Path(file).parents[1])
            if add_subfolder:
                # 4M requires files to be saved in the format modality_name/some_folder_name/file.extension
                # Here, we use subfolder as the inner folder name, but it can be anything. The one up
                # folder name must correspond to that particular modality (eg. RGB)
                train_dir = os.path.join(
                    parent_dir, "train", modality_key, "subfolder"
                )
                val_dir = os.path.join(
                    parent_dir, "val", modality_key, "subfolder"
                )
                test_dir = os.path.join(
                    parent_dir, "test", modality_key, "subfolder"
                )
            else:
                train_dir = os.path.join(parent_dir, "train")
                val_dir = os.path.join(parent_dir, "val")
                test_dir = os.path.join(parent_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            if crop_id in train_crops:
                # Save in train
                shutil.copy2(file, train_dir)
            elif crop_id in val_crops:
                # Save in val
                shutil.copy2(file, val_dir)
            elif crop_id in test_crops:
                # Save in test
                shutil.copy2(file, test_dir)
            else:
                raise ValueError(f"crop_id {crop_id} not found.")
        print(f"{modality_key} train/val/test split complete")


def dataframe_crop(
    df,
    top: int,
    left: int,
    z_dimension_length: int,
    crop_window_size: tuple,
    x_column_name: str = "x_location",
    y_column_name: str = "y_location",
    z_idx_column_name: str = "feature_name_idx",
    pixel_value_column_name: str = "pixel_value",
    dimensions: dict = {"c": -1, "y": 0, "x": 1},
):
    """
    Create an array with size crop_window_size for elements
    contained in a dataframe.

    Assumes that XY coordinate values in df have already
    been scaled to the pixel space from which the crops
    are being requested.
    """
    assert (
        len(crop_window_size) == 2
    ), f"Can only create 2D crops from a dataframe. Got {len(crop_window_size)} crop_window_size dimensions."
    bottom = top + crop_window_size[dimensions["y"]]
    right = left + crop_window_size[dimensions["x"]]

    # Filter the dataset to just contain coordinates within the crop window
    df = df.filter(
        (polars.col(x_column_name) >= left)
        & (polars.col(x_column_name) < right)
        & (polars.col(y_column_name) >= top)
        & (polars.col(y_column_name) < bottom)
    )

    # top and left define where the new origin will be for the crop
    # Now, we need to reset the pixel values in the DF to this new origin
    df = df.with_columns(
        polars.col(x_column_name).sub(left),
        polars.col(y_column_name).sub(top),
    )

    # Rearrange columns, such that
    # 0th = x_column_name,
    # 1st = y_column_name,
    # 2nd = z_idx_column_name,
    # 3rd = pixel_value_column_name
    # This standardized input to create_transcript_chunk
    # when when the DF is converted to an array
    # All other columns are dropped
    if z_idx_column_name:
        df = df[
            [
                x_column_name,
                y_column_name,
                z_idx_column_name,
                pixel_value_column_name,
            ]
        ]
    else:
        df = df[[x_column_name, y_column_name, pixel_value_column_name]]

    # Convert DataFrame to an array
    df_array = df.to_numpy()

    # Dummy output image to define the shape of the crop
    if z_idx_column_name:
        output_image = numpy.zeros(
            (*crop_window_size, z_dimension_length), dtype=int
        )
    else:
        output_image = numpy.zeros((crop_window_size), dtype=int)

    # If there are no transcripts in the crop, return an empty array
    if not df_array.size == 0:
        if z_idx_column_name:
            # df_array column order is X, Y, feature_name_idx, pixel_value
            output_image[
                numpy.round(df_array[:, 1], 0),
                numpy.round(df_array[:, 0], 0),
                numpy.round(df_array[:, 2], 0),
            ] = df_array[:, 3]
        else:
            # output_image will be 2D
            output_image[
                numpy.round(df_array[:, 1], 0),
                numpy.round(df_array[:, 0], 0),
            ] = df_array[:, 2]
        return output_image
    else:
        return output_image


def save_masked_dataframe_crops(
    dataframe: polars.DataFrame,
    image_shape: typing.Tuple,
    crop_window_size,
    z_column_name: str = "feature_name",
    save_dir: str = None,
    dimensions: dict = {"c": -1, "y": 0, "x": 1},
    filename_prefix: str = None,
    crop_scaling_factor: float = None,
    order: int = 0,
):

    crop_idx = get_crop_indices(
        image_shape=image_shape,
        crop_window_size=crop_window_size,
        dimensions=dimensions,
    )

    cropped_images = {
        f"{idx[0]}_{idx[1]}": dataframe_crop(
            dataframe,
            top=idx[0],
            left=idx[1],
            crop_window_size=crop_window_size,
            dimensions=dimensions,
            z_dimension_length=len(dataframe[z_column_name].unique()),
        )
        for idx in crop_idx
    }

    if save_dir is None:
        save_dir = "."
    else:
        os.makedirs(save_dir, exist_ok=True)

    for i, (k, crop) in enumerate(cropped_images.items()):
        if crop_scaling_factor is not None:
            # If we are rescaling, we rescale the array **and** the crop indices.
            # This allows us to later align regions between different scales
            # intuitively, since multiple scales will have the same filename crop idx.

            # Update the key with the equivalent crop indices of the resized crop
            scaled_crop_idx = (crop_idx / crop_scaling_factor).astype(int)[i]
            k = f"{scaled_crop_idx[0]}_{scaled_crop_idx[1]}"
            # Rescale the crop based on the scaling factor
            # Also, inverse the scaling since we're now scaling UP
            crop = skimage.transform.rescale(
                crop,
                scale=crop_scaling_factor**-1,
                channel_axis=-1,
                order=order,
            )
        skimage.io.imsave(
            os.path.join(save_dir, f"{filename_prefix or ''}_crop_{k}.tiff"),
            crop,
            check_contrast=False,
        )


def remove_background_crops(
    mask,
    top_and_left_idx,
    crop_window_size: typing.Iterable,
    dimensions: dict = {"c": -1, "y": 0, "x": 1},
):
    """Find the crop indices for background crops.
    These will be used to remove crops.
    """

    background_crop = array_crop(
        array=mask,
        top=top_and_left_idx[0],
        left=top_and_left_idx[1],
        crop_window_size=crop_window_size,
        padding_value=0,
        dimensions=dimensions,
    )

    if background_crop.max() == 0:
        # Return crop idx to remove
        return top_and_left_idx


def load_and_crop(
    image_or_transcript_path_dict: typing.Dict[str, str],
    scale_load_level: typing.Dict[str, int],
    crop_window_size: typing.Tuple[int, int],
    save_dir: str,
    dimensions: typing.Dict[str, int] = {"c": -1, "y": 0, "x": 1},
    convert_to_rgb: typing.Dict[str, bool] = None,
    overlapping_transcript_aggregation_method: typing.Literal[
        "mean",
        "max",
        "sum",
    ] = "mean",
    crop_idx_path: str = None,
    remove_background_crops: bool = False,
    data_to_process: typing.Dict[str, bool] = None,
    train_val_test_sizes: typing.Tuple[float, float, float] = (
        0.85,
        0.10,
        0.05,
    ),
    coordinate_space: typing.Literal["pixel", "micron"] = "pixel",
    multiscale_subset_method: typing.Literal["subset", None] = "subset",
    keep_highly_variable: bool = False,
    cell_matrix_h5_path: str = None,
    z_column_name: str = "feature_name",
    random_state: int = 42,
    gene_subset: typing.Dict[str, typing.List[str]] = None,
    cluster_id_csv_path: pandas.DataFrame = None,
    polygon_key: typing.Literal[
        "cell_boundaries", "nucleus_boundaries"
    ] = "cell_boundaries",
    non_image_modality: typing.Union[typing.List[str], str] = "transcripts",
    **kwargs,
):
    """Create crops for multiple modalities. Crops can be generated from
    image-like modalities, such as histology, and DataFrame modalities, such as
    the XY coordinates of transcript location, as found in Xenium ISS.

    Args:
        image_or_transcript_path_dict (typing.Dict[str, str]): Dictionary containing the modality name as the key and the path to the data as the value.
        Supported formats are ome.tiff, zarr, and parquet files.
        scale_load_level (typing.Dict[str, int]): For each modality name defined in image_or_transcript_path_dict, indicate the scale at which to load the
        modality. 0 represents the highest resolution, scale 1 = 2**1, scale 2 = 2**2 downsamples, etc.
        crop_window_size (typing.Tuple[int, int]): Size of the cropping window
        save_dir (str): Directory where crops will be saved.
        dimensions (_type_, optional): Dictionary indicating the index for CYX dimensions. Defaults to {"c": -1, "y": 0, "x": 1}.
        convert_to_rgb (typing.Dict[str, bool], optional): Optional conversion of any modality crop to RGB using skimage. Defaults to None.
        overlapping_transcript_aggregation_method: Method how transcript information should be downsampled. Defaults to "mean".
        crop_idx_path (str, optional): If crop indices have previously been defined, provide the path to load them. Defaults to None.
        remove_background_crops (bool, optional): Based on a provided mask, remove crops that are entirely composed of 0s in the equivalent mask crop.
        If crop_idx_path is provided, background crops will not be removed. Defaults to False.
        data_to_process (typing.Dict[str, bool], optional): Optionally choose which modalities should be processed. For transcript crop generation, a paired histology
        is required, but crops may not need to be generated. If a modality does not have a key in data_to_process, 
        it will be processed as if True. Defaults to None.
        train_val_test_sizes (typing.Tuple[float, float, float], optional): Proportion of train, val, test splits. Defaults to (0.85, 0.10, 0.05).
        coordinate_space: Transcript target coordinate space. By default, Xenium transcripts are in microns, but for image analysis we
        translate this to pixels. Defaults to "pixel".
        multiscale_subset_method: Method to subset crop indices if multiple scales are selected. Subset will randomly subset
        the crop idx to match the number of crop idx in the lowest resolution scale. None will perform no subsetting. Defaults to None.
        keep_highly_variable (bool, optional): Only keep genes that are determined to be highly variable by ScanPy. Defaults to False.
        cell_matrix_h5_path (str, optional): if keep_highly_variable=True, the transcript expression matrix path is required. Defaults to None.
        z_column_name (str, optional): Name for the feature column, which is typically the gene name for transcript crops. Defaults to "feature_name".
        random_state (int, optional): Defaults to 42.
        gene_subset (typing.Dict[str, typing.List[str]], optional): A list of gene strings to susbet to. Defaults to None.
        cluster_id_csv_path (pandas.DataFrame, optional): Path to a CSV containing cell_id and cluster ID. Used for creating cell_type maps. Defaults to None.
        polygon_key: If calculating cell types, define which boundaries will be rasterised. Defaults to "cell_boundaries".
        non_image_modality: Keys referring to non-image based modalitiees (ie. those that are not stored in an image array format), such as transcripts.
    """
    if all(v == False for v in data_to_process.values()):
        raise ValueError(
            "All data_to_process entries are False. Expected at least one True."
        )

    validate_scales(scale_load_level, non_image_modality=non_image_modality)

    # Find the scale loading levels for a representative image-based modality
    # These scale loading levels will be used later for non-image-based modality
    # realtive scaling.
    if isinstance(non_image_modality, str):
        non_image_modality = [non_image_modality]
    image_mod_keys = list(
        set(scale_load_level.keys()) - set(non_image_modality)
    )
    image_scales = {
        key: scale_load_level.get(key, "") for key in image_mod_keys
    }
    # Get the first dictionary entry of image scalings
    image_scales = next(iter(image_scales.values()))

    if save_dir is None:
        save_dir = "."
    else:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create a dictionary of lists for each modality to be loaded
    data = {k: list() for k in image_or_transcript_path_dict.keys()}
    print("Loading data...")
    # First, load the images into the data store
    for data_key, data_path in image_or_transcript_path_dict.items():
        for scale_level in scale_load_level[data_key]:
            if data_path.endswith(("ome.tiff", "ome.tif")):
                print(f"Loading {data_key} at scale {scale_level}...")
                image = load_tiff_scale(data_path, level=scale_level)

                data[data_key].append(image)

                # Get the shape of an image. This is required for transcript data scaling
                # and crop generation
                image_shape = image.shape

            elif data_path.endswith((".zarr")):
                print(f"Loading {data_key} at scale {scale_level}...")
                dask_data = load_zarr_scale(
                    data_path, level=scale_level
                )

                # Convert CYX to XYC, which is the standard in the rest of
                # histology_features
                dask_data = dask_data.swapaxes(0, 2)

                # Set the dask data to the relevant dict key
                data[data_key].append(dask_data)
                # Get the shape of an image. This is required for transcript data scaling
                # and crop generation
                image_shape = dask_data.shape

            elif data_path.endswith((".parquet")) and "transcripts" in data_key:
                if not image_shape:
                    raise ValueError(
                        "Must provide at least one image (eg. H&E or DAPI) to allow for determination of image shape for transcript crop generation"
                    )

                print(f"Loading {data_key} at scale {scale_level}...")

                transcripts = scale_and_round_xenium_transcripts(
                    df_path=data_path,
                    scale_level=scale_level,
                    # level_shape=numpy.round(image_shape / 2**image_load_level[data_key], 0).astype(int),
                    # Assumes XYC dimension order
                    level_shape=numpy.array(image_shape[:2]),
                    coordinate_space=coordinate_space,
                    overlapping_transcript_aggregation_method=overlapping_transcript_aggregation_method,
                    keep_highly_variable=keep_highly_variable,
                    cell_matrix_h5_path=cell_matrix_h5_path,
                    gene_subset=gene_subset[data_key] if gene_subset else None,
                    **kwargs,
                )

                data[data_key].append(transcripts)

                # The transcript crop output will be an array with shape (X, Y, gene_idx). {data_key}_gene_to_idx_key.npy will
                # allow for decoding of the gene index (which is an int) to a particular gene name.
                # TODO: do something better than just taking the 0th index
                gene_key = (
                    data[data_key][0][[z_column_name, f"{z_column_name}_idx"]]
                    .unique()
                    .to_numpy()
                )
                numpy.save(
                    os.path.join(save_dir, f"{data_key}_gene_to_idx_key.npy"),
                    gene_key,
                    allow_pickle=True,
                )

            elif pathlib.Path(data_path).is_dir() and "cell_type" in data_key:
                print(f"Loading {data_key} at scale {scale_level}...")

                cell_types = scale_spatialdata_polygons_and_add_cluster_id(
                    sdata_path=data_path,
                    cluster_id_csv_path=cluster_id_csv_path,
                    pixel_size=Xenium.pixel_size.value,
                    scale_level=scale_level,
                    polygon_key=polygon_key,
                )

                data[data_key].append(cell_types)

    if not crop_idx_path:
        crop_idx = {}
        print("Determining cropping indices...")
        # Define the crops from the first image encountered
        for data_name in data.keys():
            for scale_level, data_array in zip(
                scale_load_level[data_name], data[data_name]
            ):
                # We only define the crop indices from image-like modalities
                if isinstance(data_array, dask.array.Array):
                    crop_idx[f"scale_{scale_level}"] = get_crop_indices(
                        image_shape=data_array.shape,
                        crop_window_size=crop_window_size,
                        dimensions=dimensions,
                    )
            # Define the crop indices for the first image-like modality, then exit.
            # We don't need to define crop indices for each modality independently, since they
            # are all aligned spatially.
            break
        if multiscale_subset_method.casefold() == "subset":
            print("Performing crop index subsetting...")
            crop_idx = subset_equal_dict_values(
                crop_idx, random_state=random_state
            )
    else:
        print("Loading cropping indices from file...")
        crop_idx = numpy.load(crop_idx_path)
        crop_idx = {key: crop_idx_path[key] for key in crop_idx_path}

    # Find crop indices that are background. Remove these so we don't create unnecessary background crops
    # If crop idx are provided, don't find mask crops.
    if "mask" in data and remove_background_crops and not crop_idx_path:
        total_removed_crops = 0
        # Iterate over multiple mask scales
        for crop_idx_scale_name, image_scale_level, (scale_idx, mask) in zip(crop_idx.keys(), scale_load_level["mask"], enumerate(data["mask"])):
            print("Finding background crop indices...")

            # Ensure they equal the image shape
            if isinstance(mask, dask.array.Array) or isinstance(
                mask, dask.array.core.Array
            ):
                # Compute mask if a dask array
                mask = mask.compute()
            
            # TODO: fix this check for multiscale
            # for data_key in data.keys(): 
            #     # Check that the sizes match. At this point, we know
            #     if "mask" not in data_key:
            #         data_array = data[data_key][scale_idx]
            #         assert (
            #             data_array.shape[:2] == mask.shape[:2]
            #         ), f"Image and mask are not the same shape. Got image: {data_array.shape[:2]} and mask: {mask.shape[:2]}"

            # Create a numpy array in shared memory for multiprocessing
            multiprocessing_mask = numpy_to_raw_array(mask)

            find_background_crops_fn = functools.partial(
                global_remove_background_crops,
                # Only assign keywords AFTER the argument to be
                # iterated has been passed
                crop_window_size=crop_window_size,
                dimensions=dimensions,
            )
            with Pool(
                processes=cpu_count() - 1,
                initializer=init_worker,
                initargs=(multiprocessing_mask, mask.shape, mask.dtype),
            ) as pool:
                crop_idx_to_remove = list(
                    pool.map(find_background_crops_fn, crop_idx[crop_idx_scale_name])
                )
                crop_idx_to_remove = numpy.array(
                    [i for i in crop_idx_to_remove if i is not None]
                )
                print(
                    f"Deleting {len(crop_idx_to_remove)} background crops per modality at scale {image_scale_level}..."
                )
                crop_idx[crop_idx_scale_name] = remove_array_overlap(
                    crop_idx[crop_idx_scale_name], crop_idx_to_remove
                )
                total_removed_crops += len(crop_idx_to_remove)

        print(f"Background crop deletion complete. Removed a total of {total_removed_crops} background crop indices.")
        # Remove the mask data
        del data["mask"]

    # Remove modalities that aren't required for cropping
    if data_to_process is not None:
        keys_to_delete = []
        for data_name, data_array in data.items():
            # If a key is excluded from data_to_process, it will
            # be processed as if it were True.
            if not data_to_process[data_name]:
                keys_to_delete.append(data_name)
        for key in keys_to_delete:
            del data[key]

    if not crop_idx_path:
        # Save the crop indices if we haven't already
        print(
            f"Saving crop indices in {os.path.join(save_dir, 'crop_idx.npz')}"
        )
        numpy.savez(os.path.join(save_dir, "crop_idx.npz"), **crop_idx)

    print("Generating crops...")
    for data_name in data.keys():
        for scale_name, scale_level, image_scale_level, data_array in zip(
            crop_idx.keys(),
            scale_load_level[data_name],
            image_scales,
            data[data_name],
        ):
            loop_save_dir = os.path.join(save_dir, data_name)
            pathlib.Path(loop_save_dir).mkdir(parents=True, exist_ok=True)

            if isinstance(data_array, dask.array.Array):
                # Peform image cropping
                print(
                    f"Creating {crop_idx[scale_name].shape[0]} crops for {data_name} at scale {scale_level}..."
                )

                data_array = data_array.compute()
                multiprocessing_data_array = numpy_to_raw_array(data_array)

                image_crop_fn = functools.partial(
                    global_crop_array_and_save,
                    crop_window_size=crop_window_size,
                    dimensions=dimensions,
                    save_dir=loop_save_dir,
                    filename_prefix=scale_name,
                )

                with Pool(
                    processes=cpu_count(),
                    initializer=init_worker,
                    initargs=(
                        multiprocessing_data_array,
                        data_array.shape,
                        data_array.dtype,
                    ),
                ) as pool:
                    # Don't need to process the returned since we are saving inside
                    # crop_array_and_save
                    _ = list(
                        pool.imap_unordered(image_crop_fn, crop_idx[scale_name])
                    )

            elif (
                isinstance(data_array, polars.DataFrame)
                and "transcripts" in data_name
            ):
                print(
                    f"Creating {crop_idx[scale_name].shape[0]} crops for {data_name} at scale {scale_level}..."
                )
                z_dimension_length = len(data_array[z_column_name].unique())

                # Determine the relative scaling of transcriptomics to other modalities
                # No scaling difference would produce a result of 0, which 2**0 = 1
                # which would lead to crop indices and crop_window_size not changing.

                # Since we are now processing a non-image modality, we need to determine the
                # relative scaling to image-based modalities.
                # At this stage, we know image scale levels are OK.
                transcript_downsample_factor = scale_level - image_scale_level

                # Polars itself uses all cores, so performing
                # multiprocessing leads to hangups.
                # See: https://docs.pola.rs/user-guide/misc/multiprocessing/
                # So, just iterate
                for ci in crop_idx[scale_name]:
                    crop_dataframe_and_save(
                        data_array,
                        top_and_left_idx=ci,
                        crop_window_size=crop_window_size,
                        transcript_downsample_factor=transcript_downsample_factor,
                        dimensions=dimensions,
                        save_dir=loop_save_dir,
                        z_dimension_length=z_dimension_length,
                        filename_prefix=scale_name,
                    )

            elif (
                isinstance(data_array, geopandas.GeoDataFrame)
                and "cell_types" in data_name
            ):
                # Partial geopandas crop function
                subet_fn = functools.partial(
                    create_geopandas_crops,
                    data_array,
                    crop_window_size=crop_window_size,
                )

                # Make the geopandas dataframe crops
                print(f"Making geopandas subsets for scale {scale_level}...")
                with Pool(
                    processes=cpu_count(),
                ) as pool:
                    geodataframe_crops = list(
                        pool.map(subet_fn, crop_idx[scale_name])
                    )

                print(
                    f"Creating crops for {data_name} at scale {scale_level}..."
                )
                rasterize_fn = functools.partial(
                    rasterize_geometries_and_save,
                    crop_window_size=crop_window_size,
                    save_dir=loop_save_dir,
                    filename_prefix=scale_name,
                )

                with Pool(
                    processes=cpu_count(),
                ) as pool:
                    _ = list(
                        pool.imap_unordered(rasterize_fn, geodataframe_crops)
                    )

    for data_name, to_convert in convert_to_rgb.items():
        if to_convert:
            print(f"Converting {data_name} to RGB...")
            crop_path = os.path.join(save_dir, data_name)
            tiff_files = [
                os.path.join(crop_path, file) for file in os.listdir(crop_path)
            ]
            with Pool(cpu_count()) as pool:
                pool.map(load_gray2rgb_save, tiff_files)

    # Train test split
    print("Performing train/val/test split...")
    modality_crop_paths = {
        k: os.path.join(v, k)
        for k, v in zip(data.keys(), [save_dir] * len(data))
    }
    train_test_split_crops(
        modality_crop_paths,
        train_val_test_sizes=train_val_test_sizes,
        random_state=random_state,
    )
    print("Done")


def init_worker(input_array, input_array_shape, input_array_dtype):
    """Creates global variables for shared arrays"""
    global array, array_shape, array_dtype
    array, array_shape, array_dtype = (
        input_array,
        input_array_shape,
        input_array_dtype,
    )


def global_remove_background_crops(*args, **kwargs):
    # Convert the shared array to a numpy object and reshaped as desired
    shared_array = numpy.frombuffer(array, dtype=array_dtype).reshape(
        array_shape
    )
    return remove_background_crops(
        shared_array,  # Global variable
        # np.frombuffer(array).reshape(array_shape)
        *args,
        **kwargs,
    )


def global_crop_array_and_save(*args, **kwargs):
    # Convert the shared array to a numpy object and reshaped as desired
    shared_array = numpy.frombuffer(array, dtype=array_dtype).reshape(
        array_shape
    )
    return crop_array_and_save(
        shared_array,
        *args,
        **kwargs,
    )


def validate_scales(
    scale_dict: typing.Dict[str, typing.List[int]],
    non_image_modality: typing.Union[typing.List[str], str] = "transcripts",
):
    """Check that all modalities have:

    1. The same number of scales
    2. All image-based modalities have the same scales
    3. Any non-image modalities have scales that are equal or less
    than the maximum image-modality scale.

    """
    if isinstance(non_image_modality, str):
        non_image_modality = [non_image_modality]

    lengths = [len(v) for _, v in scale_dict.items()]
    assert all(
        length == lengths[0] for length in lengths
    ), "Modalities must all have the same number of scale levels."

    # Define keys associated with image-like modalities
    image_mod_keys = list(set(scale_dict.keys()) - set(non_image_modality))

    # Create modality type dictionary subsets
    image_scales = {key: scale_dict.get(key, "") for key in image_mod_keys}
    non_image_scales = {
        key: scale_dict.get(key, "") for key in non_image_modality
    }

    assert all(
        item == list(image_scales.values())[0] for item in image_scales.values()
    ), "All image-based modalities must have the same scaling levels."

    # Find all of the image scales
    all_image_scales = set(
        [scale for scale_list in image_scales.values() for scale in scale_list]
    )
    all_non_image_scales = set(
        [
            scale
            for scale_list in non_image_scales.values()
            for scale in scale_list
        ]
    )

    assert all(
        [i >= max(all_image_scales) for i in all_non_image_scales]
    ), "Non-image modality scales must be equal or less than the maximum scale of an image-based modality"


def subset_equal_dict_values(dictionary, random_state: int = 42):
    """Subset values in a dictionary

    1. Get length of all values
    """

    numpy.random.seed(seed=random_state)

    smallest_num = min([arr.shape[0] for arr in list(dictionary.values())])

    for k, v in dictionary.items():
        dictionary[k] = v[
            numpy.random.choice(v.shape[0], smallest_num, replace=False), :
        ]

    return dictionary
