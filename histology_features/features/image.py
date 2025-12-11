import os
import typing
from enum import Enum
from pathlib import Path

import cv2
import dask.array
import dask_image.ndinterp
import numpy
import scipy
import skimage
import tifffile
import zarr
import einops
import shutil

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_image

from ..features.transcripts import scale_and_round_xenium_transcripts
from ..spec.xenium import Xenium
from ..segmentation.detection import tissue_detection


class Image(Enum):
    label_idx = 0
    c = 1
    t = 2
    z = 3
    y = 4
    x = 5


def load_tiff_scale(tiff_path: str, level: int) -> dask.array.Array:
    """From a OME-TIFF, load only a specific level. Prevents loading
    of an entire image into memory."""
    with tifffile.imread(tiff_path, aszarr=True) as store:
        # If len is 2, zarr store only contains one array
        if len(store) == 2:
            image = dask.array.from_zarr(store)
        else:
            # Otherwise, load the required array
            image = dask.array.from_zarr(store, level)
    return image

def load_zarr_scale(zarr_path: str, level: int) -> dask.array.Array:
        reader = Reader(parse_url(zarr_path))
        # Nodes can represent images, labels and more
        nodes = list(reader())
        # First node is the image pixel data
        image_node = nodes[0]

        # Load the image at the desired level
        dask_data = image_node.data[level]

        return dask_data


def get_ome_tiff_shapes(ome_tiff_path: str) -> list:
    """For a pyramidal OME tiff, read the image shapes"""
    store = tifffile.imread(ome_tiff_path, aszarr=True)
    z = zarr.open(store, mode="r")
    try:
        # There's multiple layers
        levels = [int(i) for i in z]
        shapes = [z[i].shape for i in levels]
        store.close()
        return shapes
    except TypeError:
        # There's (likely) only one layer
        return z.shape


def get_ome_tiff_levels(ome_tiff_path: str) -> int:
    """Get how many levels are in an ome tiff"""
    store = tifffile.imread(ome_tiff_path, aszarr=True)
    z = zarr.open(store, mode="r")
    levels = [int(i) for i in z]
    store.close()
    return len(levels)


def save_pyramidal_ome_tiff(
    array: numpy.ndarray,
    filename: str,
    sub_resolutions: int = 7,
    dimensions: dict = {"c": 2, "x": 1, "y": 0},
):
    """Modified from tifffile documentation.

    Stores as XYCZT dimension order, which is the order tiffile uses
    (it's also the numpy standard). However, this is different
    from the OME standard.
    """

    # if array.ndim == 3 and array.shape[-1] != 3:
    #     raise ValueError(f"Expected image with shape (y, x, 3), got {array.shape}")

    if not filename.endswith(".ome.tiff"):
        filename = filename + ".ome.tiff"

    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        options = dict(
            photometric="rgb",
            maxworkers=None,
        )
        tif.write(
            array,
            subifds=sub_resolutions,
            **options,
        )
        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level in range(sub_resolutions):
            mag = 2 ** (level + 1)
            downsampled_image = cv2.resize(
                array,
                # Tiffile uses XYCZT store
                dsize=(
                    array.shape[dimensions["x"]] // mag,
                    array.shape[dimensions["y"]] // mag,
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            tif.write(
                downsampled_image,
                subfiletype=1,
                **options,
            )

def save_dask_to_zarr(
        dask_array: dask.array.Array,
        file_name: str,
        dimensions: typing.Dict[str, int] = {"x": 0, "y": 1, "c": 2},
        overwrite: bool = True,
        chunks: typing.Tuple[int] = (3, 1024, 1024)
) -> None:
    """Save a dask array to a zarr directory.
    
    Axes will be transposed to the OME-Zarr standard
    (C, Y, X) and 5 resolution downsampled levels."""

    if Path(file_name).is_dir():
        if overwrite:
            shutil.rmtree(file_name)
        else:
            raise ValueError(f"File {file_name} already exists and overwrite is {overwrite}.")

    zarr_store = parse_url(file_name, mode="w").store
    zarr_group = zarr.group(store=zarr_store)

    dask_array = dask_array.transpose(
        [dimensions["c"], dimensions["y"], dimensions["x"]]
    )

    write_image(
            image=dask_array, 
            group=zarr_group, 
            axes=["c", "y", "x"],
            # storage_options=dict(chunks=(3, 1024, 1024)),
            )



def rgb_to_gray(image, dtype=numpy.uint8):
    """Convert an RGB image to grayscale and rescale pixel intensities
    to between [0, 1]."""
    # Convert RGB to a single channel grayscale image
    image = skimage.color.rgb2gray(image)
    # Rescale the pixel intensities from 0-1 to 0-255 (as in RGB images)
    # Also, define a dtype for the array
    image = skimage.util.img_as_ubyte(image).astype(dtype)
    return image


def scale_similarly(multiscale_feature_blocks: list, order: int = 0) -> list:
    """Scale all feature blocks to be equal to the largest"""
    max_shape = max([i.shape for i in multiscale_feature_blocks])

    multiscale_feature_blocks = [
        skimage.transform.resize(i, output_shape=max_shape, order=order)
        for i in multiscale_feature_blocks
    ]

    return multiscale_feature_blocks


# TODO: remove
def _transcript_image(
    df_path: str,
    image_path: str,
    scale_level: int,
    coordinate_space: typing.Literal["pixel", "micron"],
    sample=None,
    dimensions: dict = {"c": -1, "y": 0, "x": 1},
):
    """Generate a downsampled Xenium transcript image. Transcripts
    present in the same pixel will be summed

    coordinate_space: Transcript XY coordinates are in the **micron** coordinate
    space. If micron is selected, the output image will be rescaled into micron
    space. The negative impact of this is that histology image features are
    downsampled, as H&E is in the pixel coordinate system. To scale transcript
    coordinates to pixel, choose pixel coordinate_space
    """

    # Get the image size for the desired level without loading the image
    level_shape = get_ome_tiff_shapes(image_path)[scale_level]

    # Keep only XY
    level_shape = (level_shape[dimensions["y"]], level_shape[dimensions["x"]])

    assert (
        len(level_shape) == 2
    ), f"Image path has {len(level_shape)} dimensions, expected 2."

    if coordinate_space.casefold() == "micron":
        # Scale the image from the pixel coordinate system to the micron coordinate
        # system
        scaled_shape = numpy.round(
            numpy.array(level_shape) * Xenium.pixel_size.value
        ).astype(int)
    elif coordinate_space.casefold() == "pixel":
        # Micron transcript coordinates will be scaled to the H&E pixel
        # coorindate system
        scaled_shape = level_shape
    else:
        raise ValueError(f"coordinate_space {coordinate_space} not recognised.")

    # Subsample transcripts and scale down to desired level
    df, n_unique, encoded_labels = scale_and_round_xenium_transcripts(
        df_path,
        scale_level=scale_level,
        level_shape=scaled_shape,
        sample=sample,
        coordinate_space=coordinate_space,
    )

    # Create empty array to store transcript blocks
    transcript_image = numpy.zeros((*scaled_shape, n_unique.compute()), dtype=int)

    # Convert Dask DataFrame to an array
    df_array = df.to_dask_array()

    # Delay the chunk processing function
    delayed_create_transcript_chunk = dask.delayed()(create_transcript_chunk)

    # Delay the dask array and ravel the blocks
    blocks = df_array.to_delayed().ravel()

    # Process each block. This returns a block per darr block, so will need to
    # be summed
    results = dask.array.array(
        [
            dask.array.from_delayed(
                delayed_create_transcript_chunk(blk, transcript_image.shape),
                transcript_image.shape,
                dtype=int,
            )
            for blk in blocks
        ]
    )

    # Axis 0 is the list length dimension, so we sum all dask array chunks into
    # one.
    transcript_blocks = dask.array.sum(results, axis=0)
    transcript_blocks = transcript_blocks.compute()
    return transcript_blocks, encoded_labels


def apply_xenium_alignment(
    he_image: typing.Tuple,
    dapi_image_shape: numpy.ndarray,
    alignment_matrix: numpy.ndarray,
    level: int,
    dimensions: typing.Dict[str, int] = {"c": 2, "x": 1, "y": 0}
) -> numpy.ndarray:
    """Xenium explorer can perform key-point based image
    registration. This process outputs a 3x3 affine transformation matrix. The
    layout of transformation matrices is as follows:

    [
        [scale, rotate, translate],
        [rotate, scale, translate],
        [None, None, 1]
    ]

    scale: This value is what each coordinate in he_image will be multiplied by.
    Thus, as long as you've loaded the same level of dapi_image and he_image,
    scale does not need to be adjusted to the level

    rotate: The rotation amount in radians

    translate: How the coorindates of he_image should be moved to match that of
    dapi_image. This value will have to be rescaled to match the coordinate
    system of the image level loaded.
    """

    assert alignment_matrix.shape == (3, 3)

    # Adjust translation transformation to match the level of image loaded
    alignment_matrix[1, -1] = alignment_matrix[1, -1] / 2**level
    alignment_matrix[0, 2] = alignment_matrix[0, 2] / 2**level

    # Output shape is dapi_image since the transformation matrix is
    # translating the he_image coordinates to that of the dapi_image
    # Remember, the DAPI image is acquired when transcripts are acquired
    # so we are using the DAPI image as a proxy for transcript alignment too.
    image = dask.array.zeros((dapi_image_shape + (3,)), dtype=he_image.dtype)
    # For SciPy/dask_image affine_transform, the transform uses the "pull"
    # resampling, which is why we inverse the alignment matrix.
    # cv2 and skimage affine transforms use the "push" resampling method.
    # We also transpose the image and then transpose the result
    # back. This switches XY to YX, which is what we need for the
    # homography matrix.

    for ch in range(he_image.shape[2]):
        # image[...,ch] = scipy.ndimage.affine_transform(
        image[..., ch] = dask_image.ndinterp.affine_transform(
            he_image[..., ch].T,
            numpy.linalg.inv(alignment_matrix)[:2],
            output_shape=(dapi_image_shape[1], dapi_image_shape[0]),
        ).T

    return image


def normalise_rgb(image, mean, std):
    # Convert image to float32 to ensure precision during calculations
    image = image.astype(numpy.float32)
    # Normalize each channel separately
    for i in range(3):  # assuming RGB channels
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

    return image


def get_object_bounding_box(
    array: numpy.ndarray, pad: typing.Union[int, float] = None
) -> list:
    """For a binary image, find the bounding boxes of the objects.

    Optionally pad these bounding boxes by a fixed number of pixels by providing
    an int, or by a percentage of the major axis (height of width) by providing
    a float."""
    labels = skimage.measure.label(array)

    props = skimage.measure.regionprops(labels)

    bbox = [i["bbox"] for i in props]

    # Perform padding around the bbox
    if pad is not None:
        for i, bb in enumerate(bbox):
            min_row, min_col, max_row, max_col = bb
            if isinstance(pad, float):
                # Pad as a fraction of the bounding box major axis
                pad = round(max(pad * (max_row - min_row), pad * (max_col - min_col)))
            min_row_padded = max(min_row - pad, 0)
            min_col_padded = max(min_col - pad, 0)
            max_row_padded = min(max_row + pad, array.shape[0])
            max_col_padded = min(max_col + pad, array.shape[1])
            bbox[i] = (min_row_padded, min_col_padded, max_row_padded, max_col_padded)

    return bbox


def save_sequential_crops(
    image_directory: str,
    mask_bbox_padding: typing.Union[int, float],
    image_level: int,
    mask_level: int,
    transposition: typing.Tuple = None,
    inverse_label_order: bool = False,
    min_region_size: int = 5_000,
    max_hole_size: int = 1_500,
    outer_contours_only: bool = False,
    blur_kernel: int = 17,
    morph_kernel_size: int = 7,
    morph_n_iterations: int = 3,
    manual_threshold: int = None,
) -> None:
    """For images in a directory, threshold the image and crop the image based on
    bounding boxes. Save as a 6 level pyramidal ome.tiff

    Images should be in the shape (y, x, c). If they are not, include a transposition
    tuple. 

    Args:
        image_directory (str): Directory that contains ome.tiff images
            with multiple ROIs in single images
        mask_bbox_padding (typing.Union[int, float]): Mask proportion of padding
            to apply to the output crop. Prevents ROIs being too close to the edge
            of an image
        image_level (int): Subresolution level of image that will be cropped
        mask_level (int): Subresolution level at which to calculate the mask. The
            lower the resolution, the faster mask calculation will be
        inverse_label_order (bool, optional): By default, saved ROIs are
            labelled from right to left as they appear in the image.
            set this to True to make the numbering left to right. Defaults to False.
    """

    if not os.path.exists(image_directory):
        raise ValueError(f"Directory {image_directory} does not exist.")

    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.endswith((".ome.tiff", "ome.tif")):
                print(f"Processing {file}")
                file_to_load = os.path.join(root, file)
                # Load the image to be cropped
                img = load_tiff_scale(file_to_load, level=image_level).compute()
                if transposition:
                    img = img.transpose(transposition)

                # Calculate the mask at a lower resolution since we just need
                # a rough idea of where the sample iss
                img_to_threshold = load_tiff_scale(
                    file_to_load, level=mask_level
                ).compute()
                if transposition:
                    img_to_threshold = img_to_threshold.transpose(transposition)

                # Get the low resolution mask. Faster to compute and also allows
                # for only coarse foreground to be considiered (ie. not very small objects)
                mask = tissue_detection(
                    img_to_threshold, 
                    min_region_size=min_region_size,
                    max_hole_size=max_hole_size,
                    outer_contours_only=outer_contours_only,
                    blur_kernel=blur_kernel,
                    morph_kernel_size=morph_kernel_size,
                    morph_n_iterations=morph_n_iterations,
                    manual_threshold=manual_threshold,
                )
                # Resize the lower resolution image to the full size
                mask = skimage.transform.resize(mask, img.shape[:2], order=0)

                bboxes = get_object_bounding_box(mask, pad=mask_bbox_padding)

                if inverse_label_order:
                    bboxes.reverse()

                # Make a cropping subdirectory
                os.makedirs(os.path.join(image_directory, "crops"), exist_ok=True)

                # Crop the full resolution image for each bounding box
                for i, bb in enumerate(bboxes):
                    save_filename = os.path.join(
                        image_directory,
                        "crops",
                        f"{Path(file_to_load).stem}_roi_{i}.ome.tiff",
                    )
                    print(f"Saving crop number {i}")
                    cropped_img = img[bb[0] : bb[2], bb[1] : bb[3]]
                    save_pyramidal_ome_tiff(array=cropped_img, filename=save_filename)

        # Prevent os.walk recursiveness with a break. We only want one level
        break

def zarr_to_pyramidal_ome_tiff(zarr_path, output_tiff_path, include_metadata=False):
    """
    Convert a Zarr directory into a pyramidal OME-TIFF file (with multiple resolution levels).
    
    Parameters:
    - zarr_path (str): Path to the Zarr directory.
    - output_tiff_path (str): Path to save the OME-TIFF file.
    - include_metadata (bool): Whether to include OME-XML metadata.
    
    Returns:
    - None

    TODO: pyramidal saving does not work for this function.
    """
    
    store = parse_url(zarr_path)
    reader = Reader(store)
    nodes = list(reader())
    if not nodes:
        raise ValueError("No data found in the Zarr store")
    
    multiscale = nodes[0]
    pyramid = multiscale.data
    
    ome_metadata = None
    if include_metadata:
        # Use the highest resolution (pyramid[0]) to create the base metadata
        size_z, size_y, size_x = pyramid[0].shape
        image_metadata = Image(
            name="Pyramidal OME-TIFF",
            pixels=Pixels(
                dimension_order="ZYX",
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                size_c=1,  # Adjust based on your data
                size_t=1,  # Adjust based on your data
                type=str(pyramid[0].dtype),
            )
        )
        ome_metadata = image_metadata.to_xml()
    
    with tifffile.TiffWriter(output_tiff_path, bigtiff=True) as tif:
        for level, resolution_data in enumerate(pyramid):
            data = np.array(resolution_data)  # Convert Dask array to NumPy array if necessary
            print(level, data.shape)
            tif.write(
                data,
                photometric='minisblack',
                metadata={'axes': 'ZYX'} if not include_metadata else {'axes': 'ZYX', 'Description': ome_metadata},
                # resolution=(1.0, 1.0),  # You can adjust this for pixel size at each resolution level
                subfiletype=0 if level == 0 else 1  # 0 for the base image, 1 for reduced resolutions
            )
    
    print(f"Pyramidal OME-TIFF file saved to {output_tiff_path}")