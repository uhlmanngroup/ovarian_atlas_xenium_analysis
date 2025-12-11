import typing

import dask.array
import mahotas
import numpy
import skimage
import torch


def lbp_features(
    image: numpy.ndarray,
    radius: typing.Union[list, int],
):
    """Extract LBP features from an image. The 'footprint' of the LBP features
    can be controlled with the radius parameter. Multiple radii provided will
    lead to multiple scales being recorded"""
    if isinstance(radius, int):
        radius = [radius]

    output = numpy.zeros((len(radius), *image.shape))

    for i, rad in enumerate(radius):
        n_points = numpy.pi * rad**2
        output[i, ...] = skimage.feature.local_binary_pattern(
            image, R=rad, P=n_points, method="nri_uniform"
        )
    # Flatten all features into a single vector
    output = output.ravel()
    # # Reshape output to have the same ndim as the input. This is a requirement
    # # for Dask map_blocks, which expects the input and output ndim to match.
    # # Empty dims work for this, so we just insert new dims at the 0th position,
    # # which will later correspond to the XY coords of the block a feature
    # # corresponds to.
    # output = expand_dims_by(output, len(image.shape) - len(output.shape))

    return output


def lbp_features2(
    image: numpy.ndarray, radius: typing.Union[list, int], radius_multiplier: int = 8
) -> numpy.ndarray:
    """Extract LBP features from an image. The 'footprint' of the LBP features
    can be controlled with the radius parameter. Multiple radii provided will
    lead to multiple scales being recorded.

    Better implementation, use this one."""
    if isinstance(radius, int):
        radius = [radius]

    lbp_hist = numpy.array([])

    for i, rad in enumerate(radius):
        if radius_multiplier is not None:
            n_points = radius_multiplier * rad
        else:
            n_points = numpy.pi * rad**2

        output = skimage.feature.local_binary_pattern(
            image, R=rad, P=n_points, method="uniform"
        )
        hist = numpy.histogram(
            output.ravel(), bins=numpy.arange(0, n_points + 3), range=(0, n_points + 2)
        )[0]
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-7
        lbp_hist = numpy.concatenate([lbp_hist, hist])

    return lbp_hist


def hog_features(
    image: typing.Union[numpy.ndarray, dask.array.Array],
) -> typing.Union[numpy.ndarray, dask.array.Array]:
    """Extract HoG features from an image"""
    hog = skimage.feature.hog(image)
    hist = numpy.histogram(hog, bins=40)[0]
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-7

    return hist


def haralick_features(image: numpy.ndarray, error_value: int = 1):
    """Extract haralick features from an image.

    TODO: fix error encountered when an empty image (ie all 0s) is
    encountered"""
    try:
        features = mahotas.features.haralick(image).mean(axis=0)
        features = features.ravel()
    except ValueError:
        # Haralick struggles to handle empty input
        # Catch this, and return an empty feature vector
        # features = numpy.ones((13,))
        features = numpy.full((13,), fill_value=error_value)
    return features


def vision_transformer_features(
    image: torch.Tensor, model, transform, dict_result_key: str = "last_hidden_state"
):
    """Apply a transformer model (assumed to be a HuggingFace transformer) to an
    image. **Returns the class embedding**"""

    image = transform(image)
    # Add batch dim
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)[dict_result_key].numpy()

    # Take 0th feature since this is the class embeddings, rather
    # than individual patch embeddings
    features = features[0, 0, ...]

    return features


def transformer_features(image: numpy.ndarray, model, transforms_to_apply):
    """Function that applies a HuggingFace transformer to
    an image. Specifically, this is intended to be used with dask blocks.

    Each dask block is transformed upon compute.

    Both patch and block feature vectors are returned.

    The return array is peculiar when ran with dask, so must be post-processed
    using `postprocess_dask_transformer_features`
    """
    import torchvision
    import torchvision.transforms as transforms

    # Apply transformations
    if transforms_to_apply is not None:
        image = transforms_to_apply(image)
    else:
        assert isinstance(image, numpy.ndarray), "Input must be a numpy array"
        # No transforms passed, so convert to Tensor
        transforms_to_apply = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        image = transforms_to_apply(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = image.to(device)
    model.to(device)

    # Add batch dimension
    image = image.unsqueeze(0)

    output = model(image)

    # Get the patch features
    patch_features = output.last_hidden_state[:, 1:, :].squeeze().cpu().detach().numpy()
    # Infer how many patches in one dimension
    patch_dim = numpy.sqrt(patch_features.shape[0]).astype(int)
    patch_features = patch_features.reshape(patch_dim, patch_dim, -1)

    # Get the whole image features
    image_features = output.last_hidden_state[:, 0, :].squeeze().cpu().detach().numpy()

    return numpy.array([patch_features, image_features], dtype="object")


def postprocess_dask_transformer_features(transformer_features: numpy.ndarray):
    """Process the output of transformer_features into a feature_block image"""

    if transformer_features.ndim == 2:
        raise ValueError(
            f"Are you sure feature blocks were extracted for this image? Detected {transformer_features.ndim}, expected 4"
        )

    # Determine what the original block dims were
    block_dims = numpy.array(transformer_features[..., 0].squeeze().shape)
    # Get the length of the feature vector
    feature_vector_length = transformer_features[..., 0].squeeze()[0, 0].shape[-1]

    ### Reshape patch features into feature block image
    # Determine what the patch shape will be. Ie. a 1x1 block will have a shape
    # 14x14, since there are 14 patches per 224x224 image.
    patch_dims = numpy.array(transformer_features[..., 0].squeeze()[0, 0].shape[:-1])

    # First, let's flatten our nested arrays. eg. this convers (2, 3) into (6,)
    # Each of the 6 arrays has shape (14, 14, feature_vec)
    patches = transformer_features[..., 0].squeeze().flatten()

    # Now, we can stack the flattened patch arrays. This homogenises the arrays
    # and doesn't lead to nested and object arrays, which is an outcome of Dask
    # computing the ragged feature blocks.
    patches = numpy.stack(patches)

    # Now, we will need to reshape. However, it's not a straightforward
    # operation Eg. we have 6 "layers" of (14, 14, feature_vector) that need to
    # be reshaped in the same manner they were generated. Ie. 0th layer is in
    # the [0, 0] position. 1th layer is in the [0, 1] position. So, we will
    # perform a reshape, swapaxes, reshape First, break up 6 into (row_idx,
    # col_idx), which is (2, 3)
    patches = patches.reshape(*block_dims, *patch_dims, feature_vector_length)

    # Next, we swapaxes so that it's (row_idx, patch_dim, col_idx, patch_dim,
    # feature_vec)
    patches = patches.swapaxes(1, 2)

    # Perform the final reshape with the patches in the correct order
    patches = patches.reshape(*patch_dims * block_dims, feature_vector_length)

    ### Reshape "CLS" token features into feature block image
    cls_features = numpy.stack(
        transformer_features[..., 1].squeeze().flatten()
    ).reshape(*block_dims, feature_vector_length)

    return patches, cls_features


def expand_dims_by(array: numpy.ndarray, number_of_dims: int):
    """I couldn't work out a way to do this without iterating...How irritating"""
    for _ in range(number_of_dims):
        array = array[numpy.newaxis]

    return array


def get_vit_mae_features(image):
    import transformers
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    if isinstance(image, dask.array.core.Array):
        image = image.compute()

    if image.shape[0] == 1:
        image = skimage.color.gray2rgb(image.squeeze(), channel_axis=0)
    elif image.shape[0] == 3:
        # Likely a image that was stored by SpatialData. Tranpose
        # to the (x/y, y/x, c) dimension
        image = image.transpose(1, 2, 0)

    config = transformers.ViTMAEConfig.from_pretrained("facebook/vit-mae-base")
    # Since it's pretrained, we don't need to mask
    config.update({"mask_ratio": 0})
    model = transformers.ViTMAEForPreTraining.from_pretrained(
        "facebook/vit-mae-base", config=config
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((224, 224)),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    model.to(device)

    features = model.vit(image)["last_hidden_state"]
    return features.detach().cpu().numpy().ravel()
