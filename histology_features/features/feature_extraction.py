import typing
import warnings

import anndata
import dask.array
import numpy
import skimage

from .classification import apply_pixel_classifier


def feature_map_blocks(
    image, feature_extractor, window_size: typing.List, dtype=numpy.float16, **kwargs
):
    """Apply a function to a dask array that returns a different shape than
    input.

    Use case for this is applying a feature extractor over an image that returns
    a feature vector for each dask block.

    output shape will be (block_shape_0, block_shape_1, feature_vector). This
    output shape can be reshaped into something some 2D vector for all blocks
    """

    assert isinstance(window_size, list), "window_size must be a list"

    assert image.ndim == len(window_size), "Window ndim must be equal to image ndim"

    if image.ndim == 2 and len(window_size) == 2:
        # Grayscale image. Add an empty third dimension
        # to make map_blocks return vectors with shape
        # (x_pos, y_pos, feature_vector)
        image = image[..., numpy.newaxis]

        # Update the window size accordingly
        window_size.append(1)

    # map_blocks chunk output
    chunks = [1] * len(window_size)

    dask_image = dask.array.from_array(image, chunks=window_size)

    features = dask.array.map_blocks(
        feature_extractor,
        dask_image,
        dtype=dtype,
        chunks=chunks,
        **kwargs,  # Specific arguments to feature_extractor
    )

    features = features.compute()

    return features


def feature_map_blocks2(
    image: typing.Union[numpy.ndarray, dask.array.Array],
    feature_extract_function: typing.Callable,
    window_size: typing.List,
    feature_blocks: bool = False,
    mask: numpy.ndarray = None,
    **kwargs,
) -> numpy.ndarray:
    """
    Apply a function to a dask array that returns a different shape than input.

    Use case for this is applying a feature extractor over an image that returns
    a feature vector for each dask block.

    output shape will be (block_shape_x, block_shape_y, block_shape_z[optional],
    feature_vector). In other words (x_block_idx, y_block_idx,
    z_block_idx[optional], flat_feature_vector)
    """

    if image.shape[:2] == window_size[:2]:
        raise ValueError(
            f"image shape {image.shape} == window_size {window_size.shape}. Just extract features directly using {feature_extract_function} since blocks are not required."
        )

    if mask is not None:
        assert (
            mask.shape[:2] == image.shape[:2]
        ), "Mask must have the same shape as the input image"

    if (
        image.shape[0] % window_size[0] != 0
        or image.shape[1] % window_size[1] != 0
        and mask is not None
    ):
        warnings.warn(
            f"""Window size {window_size} does not divide into image size {image.shape}.
                      Window size will be automatically changed to fit by Dask, which can result in 
                      inaccurate mask resizing when clustering feature blocks."""
        )

    # Delayify the function
    feature_extract_function = dask.delayed()(feature_extract_function)

    if isinstance(image, numpy.ndarray):
        # Chunk the input image to the desired window size
        dask_image = dask.array.from_array(image, chunks=window_size)
        if mask is not None:
            # Masks are 2D, so skip the 2nd dimension, which is the channel
            dask_mask = dask.array.from_array(mask, chunks=window_size[:2])
    elif isinstance(image, dask.array.Array):
        dask_image = dask.array.rechunk(image, chunks=window_size)
        if mask is not None:
            dask_mask = dask.array.rechunk(mask, chunks=window_size[:2])
    else:
        raise NotImplementedError(f"Image of type {type(image)} is not supported.")

    # Create a function that optionally thresholds dask blocks
    # based upon a mask
    if mask is not None:
        # For each dask_image block, apply the feature extraction function
        # Only feature extract where there exists mask-defined foreground
        # ie. if an array contains any non-zero elements, run feature_extract_function
        features = dask.compute(
            [
                threshold_skip_dask_blocks(
                    dask_block, mask_block, feature_extract_function, **kwargs
                )
                for (dask_block, mask_block) in zip(
                    dask_image.to_delayed().ravel(),
                    dask_mask.to_delayed().ravel(),
                )
            ]
        )[0]
        # return features
        features = array_homogeniser(features)
        features = numpy.array(features)
    else:
        # For each dask_image block, apply the feature extraction function
        features = numpy.array(
            dask.compute(
                [
                    feature_extract_function(x, **kwargs)
                    for x in dask_image.to_delayed().ravel()
                ]
            )
        )

    # List to store the results
    output = []

    output.append(features)
    output = numpy.array(output)

    # Reshape the output so that
    # (block_shape_x, block_shape_y, block_shape_z[optional], feature_vector)
    if feature_blocks:
        # Features already exists in axis=-1, so we do not need to expand output to hold
        # feature vector
        output = output.reshape(dask_image.blocks.shape)
    else:
        # output.shape[-1] corresponds to the length of the feature vector, so
        # add this in the 4th dimension
        output = output.reshape((*dask_image.blocks.shape, output.shape[-1]))

    return output


def feature_map_overlap_blocks(
    image: typing.Union[numpy.ndarray, dask.array.Array],
    feature_extract_function: typing.Callable,
    window_size: typing.List,
    overlap: typing.Union[int, typing.Tuple[int, int]],
    boundary: typing.Union[
        typing.Literal["reflect", "periodic", "nearest", "none"], int
    ] = "reflect",
    feature_blocks: bool = False,
    mask: typing.Union[numpy.ndarray, dask.array.Array] = None,
    **kwargs,
) -> typing.Union[numpy.ndarray, dask.array.Array]:
    """Apply a function to an array with each block overlapping.

    Behind the scenes, this uses Dask overlap, but is different in that it can
    return an array of a different shape (eg feature block).

    Warning: internally, Dask will change the window size if it is not perfectly
    compatibile with the image width (ie. the length of a dimension is not
    perfectly captured by the overlapping windows)"""

    if mask is not None:
        assert (
            mask.shape[:2] == image.shape[:2]
        ), "Mask must have the same shape as the input image"

    # Define the overlap
    if not isinstance(overlap, list):
        overlap = [overlap]

    if len(overlap) != image.ndim:
        overlap = overlap * image.ndim

    for i, (ovrlp, dim) in enumerate(zip(overlap, window_size)):
        if ovrlp < 0 or ovrlp >= 1:
            raise ValueError(
                f"Overlap must be overlap < 0 or overlap >= 1, but got {ovrlp}"
            )
        # Since overlap is a percentage, multuply by the window size, in pixels,
        # to get the amount of window overlap desired
        # eg. 0.5 x 10 = 5px overlap (or in Dask terms, depth)
        overlap[i] = int(ovrlp * dim)

    # Add to a dictionary for each axis, as Dask desires
    overlap = {k: v for k, v in zip(numpy.arange(image.ndim), overlap)}

    # Delayify the function
    feature_extract_function = dask.delayed()(feature_extract_function)

    if isinstance(image, numpy.ndarray):
        # Chunk the input image to the desired window size
        dask_image = dask.array.from_array(image, chunks=window_size)
        if mask is not None:
            # Masks are 2D, so skip the 2nd dimension, which is the channel
            dask_mask = dask.array.from_array(mask, chunks=window_size[:2])
    elif isinstance(image, dask.array.Array):
        dask_image = dask.array.rechunk(image, chunks=window_size)
        if mask is not None:
            dask_mask = dask.array.rechunk(mask, chunks=window_size[:2])
    else:
        raise NotImplementedError(f"Image of type {type(image)} is not supported.")

    # Rechunk so all overlapping windows fit in dask array blocks
    dask_image = overlap_rechunk(dask_image, overlap)
    if mask is not None:
        dask_mask = overlap_rechunk(dask_mask, overlap)

    # Here, we don't allow rechunk so that the minimum chunk size is the overlap
    # requested (in Dask terms overlap = depth)
    overlapped_blocks = dask.array.overlap.overlap(
        dask_image, depth=overlap, boundary=boundary, allow_rechunk=False
    )
    if mask is not None:
        overlapped_mask_blocks = dask.array.overlap.overlap(
            dask_mask, depth=overlap, boundary=boundary, allow_rechunk=False
        )

    delayed_image_blocks = overlapped_blocks.to_delayed().ravel()

    if mask is not None:
        delayed_mask_blocks = overlapped_mask_blocks.to_delayed().ravel()
        features = dask.compute(
            [
                threshold_skip_dask_blocks(
                    dask_block, mask_block, feature_extract_function, **kwargs
                )
                for (dask_block, mask_block) in zip(
                    delayed_image_blocks,
                    delayed_mask_blocks,
                )
            ]
        )[0]
        features = array_homogeniser(features)
    else:
        features = dask.compute(
            [feature_extract_function(x, **kwargs) for x in delayed_image_blocks]
        )

    features = numpy.array(features)

    features = features[numpy.newaxis]

    # Reshape the features so that
    # (block_shape_x, block_shape_y, block_shape_z[optional], feature_vector)
    if feature_blocks:
        # Features already exists in axis=-1, so we do not need to expand features to hold
        # feature vector
        features = features.reshape(dask_image.blocks.shape)
    else:
        # features.shape[-1] corresponds to the length of the feature vector, so
        # add this in the 4th dimension
        features = features.reshape((*dask_image.blocks.shape, features.shape[-1]))

    return features


def multichannel_apply_fn(
    image: numpy.ndarray,
    feature_extraction_fn: typing.Callable,
    channel_axis: int,
    **kwargs,
) -> numpy.ndarray:
    """Apply a function along a given axis in an image.

    Basically, it's map along an axis. Perhaps it's better to use that..."""
    output = numpy.stack(
        [
            feature_extraction_fn(image.take(i, axis=channel_axis), **kwargs)
            for i in range(image.shape[channel_axis])
        ],
        axis=-1,
    )

    # Flatten output. Haven't worked out a way
    # to preserve channel specific features yet
    output = output.ravel()

    return output


def classifier_segmentation_blocks(
    image: typing.Union[numpy.ndarray, dask.array.Array],
    window_size: list,
    feature_extract_function: typing.Callable,
    classifier,
    remove_holes_threshold: int = 64,
    small_objects_threshold: int = 64,
    **kwargs,
) -> dask.array.Array:
    """
    Apply a classifier that segments an image in some pre-trained way. For
    example, segmenting foreground and background.

    Returns the feature blocks used for classifier segmentation and the
    segmentation mask for said segmentation blocks.
    """
    # Generate feature blocks
    feature_blocks = feature_map_blocks2(
        image,
        feature_extract_function=feature_extract_function,
        window_size=window_size,
        **kwargs,
    )

    # Apply a classifier to feature blocks
    mask = feature_map_blocks2(
        feature_blocks,
        feature_extract_function=apply_pixel_classifier,
        feature_blocks=True,
        window_size=[1, 1, -1],
        classifier=classifier,
        # class_mappings=class_mappings
    )

    # Tidy up the thresholded image
    if remove_holes_threshold:
        mask = skimage.morphology.remove_small_holes(
            mask, area_threshold=remove_holes_threshold
        )
    if small_objects_threshold:
        mask = skimage.morphology.remove_small_objects(
            mask, min_size=small_objects_threshold
        )

    return feature_blocks, mask


def classifier_segmentation_overlap_blocks(
    image: typing.Union[numpy.ndarray, dask.array.Array],
    window_size: list,
    overlap: int,
    feature_extract_function: typing.Callable,
    classifier,
    remove_holes_threshold: int = 64,
    small_objects_threshold: int = 64,
    **kwargs,
) -> dask.array.Array:
    """
    For a given image, apply a classifier with overlapping feature blocks.

    It's assumed the classifier is a foreground-background classifier, so the
    holes in the foreground and small objects are removed.

    TODO: Currently, this function returns the non-overlapping blocks, which is
    wrong (though their use isn't wrong, since typically used RandomForest model
    was trained using non-overlapping).
    """
    # Generate feature blocks
    feature_blocks = feature_map_blocks2(
        image,
        feature_extract_function=feature_extract_function,
        window_size=window_size,
        **kwargs,
    )

    # Apply a classifier to feature blocks
    mask = feature_map_overlap_blocks(
        feature_blocks,
        feature_extract_function=apply_pixel_classifier,
        feature_blocks=True,
        window_size=[1, 1, -1],
        overlap=overlap,
        classifier=classifier,
        # class_mappings=class_mappings
    )

    # Tidy up the thresholded image
    mask = skimage.morphology.remove_small_holes(
        mask, area_threshold=remove_holes_threshold
    )
    mask = skimage.morphology.remove_small_objects(
        mask, min_size=small_objects_threshold
    )

    return feature_blocks, mask


def overlap_rechunk(image: dask.array.Array, depths: dict) -> dask.array.Array:
    """This function is taken from dask.array.overlap
    and allows us to replace usage of the argument allow_rechunk=True in
    dask.array.overlap(). This means we can know **before** running overlap what
    the rechunk is going to be, rather than trying to work it out after running
    overlap block compute.

    This function rechunks a dask array so that the window size and overlap is
    maintained
    """
    depths = [d for d in depths.values()]
    # rechunk if new chunks are needed to fit depth in every chunk
    new_chunks = tuple(
        dask.array.overlap.ensure_minimum_chunksize(size, c)
        for size, c in zip(depths, image.chunks)
    )
    image1 = image.rechunk(new_chunks)  # this is a no-op if x.chunks == new_chunks

    return image1


# TODO: remove
def _create_transcript_chunk(
    array: dask.array.Array,
    output_shape: tuple,
    x_idx=0,
    y_idx=1,
    feature_name_idx=2,
    qv_idx=3,
):
    """This function is called when binning xenium transcript points into pixel
    coordinate space.

    Generates a numpy array using dask array of indices.

    Here, we use numpy, since Dask does not support the fancy style of indexing
    used."""
    output_img = numpy.zeros(output_shape, dtype=int)

    output_img[
        array[:, y_idx],
        array[:, x_idx],
        array[:, feature_name_idx],
    ] = array[:, qv_idx]

    return output_img


def feature_blocks_to_anndata(
    feature_blocks: numpy.ndarray,
    mask: numpy.ndarray = None,
    add_spatial: bool = False,
    spatial_key: str = "spatial",
) -> anndata.AnnData:
    """Convert a feature blocks image into a AnnData object. If mask
    is provided, masked out blocks **will not** be added to the anndata
    object"""

    # Target shape is (n_blocks, features)
    target_shape = (
        feature_blocks.shape[0] * feature_blocks.shape[1],
        feature_blocks.shape[-1],
    )

    # Reshape to array of (num_blocks, num_features_per_block)
    reshaped_features = numpy.reshape(feature_blocks, target_shape)

    if mask is not None:
        assert mask.dtype == bool, "Mask must be boolean"
        # Reshape the mask to the same format as reshaped_features, but without
        # the feature vector
        reshaped_mask = numpy.reshape(
            mask, (feature_blocks.shape[0] * feature_blocks.shape[1], 1)
        )
        # Define indices that are not masked
        idx_to_keep = numpy.where(reshaped_mask)[0]
    else:
        # No mask provided, so all blocks will be clustered
        idx_to_keep = numpy.arange(0, reshaped_features.shape[0])

    adata = anndata.AnnData(reshaped_features)[idx_to_keep]

    if add_spatial:
        block_idx = numpy.array(list(numpy.ndindex(feature_blocks.shape[:2])))
        # Keep only the unmasked
        block_idx = block_idx[idx_to_keep]
        adata.obsm[spatial_key] = block_idx

    return adata


def threshold_skip_dask_blocks(
    dask_block, mask_block, feature_extract_fn: typing.Callable, **kwargs
):
    """If a dask block is composed of entirely value 0 pixels,
    skip feature extraction and just return an empty dask array"""

    if mask_block.compute().max() == 0:
        return numpy.empty(0)

    return feature_extract_fn(dask_block, **kwargs)


def array_homogeniser(feature_blocks: list) -> numpy.ndarray:
    """
    If an array contains dimensions of different lengths, pad the
    shorter dimensions to make them to the maximum.

    This function finds the masked out feature blocks and expands
    them to match the shape of your non-masked feature blocks.

    This allows for a homogenous array of feature blocks to be created,
    thus preserving the spatial information encoded by the blocks
    while also speeding up computation by not feature extracting
    masked out blocks.

    feature_blocks: the return of dask.compute()[0], which should be
    a list of feature blocks with len(number of blocks)
    """
    # Determine feature vector shapes
    shapes = []
    for block in feature_blocks:
        # Skip empty blocks, which have been masked out
        if block.size != 0:
            # Check that a block does not have multiple feature vectors,
            # as is the case for ViT features (patch and CLS features)
            for feature_vector in block:
                shapes.append(feature_vector.shape)

    # key in set to preserve order
    # Relevant if a block has two feature vectors and
    # their index needs to be the same for all blocks
    shapes = sorted(set(shapes), key=shapes.index)

    # Now set the empty features
    for block_idx, block in enumerate(feature_blocks):
        if len(block) == 0:
            # At an empty feature block, set the feature vector
            # to be a zero array of the same shape as the other
            # foreground feature vectors
            feature_blocks[block_idx] = numpy.array(
                [numpy.zeros(i) for i in shapes], dtype="object"
            )

    return feature_blocks
