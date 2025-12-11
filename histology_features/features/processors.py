"""
File for pipelining functions. 

ie. A function that stitches together a series of other 
commonly used functions.
"""

import scipy
import skimage
import torchvision.transforms as transforms

import histology_features
# from histology_features.features.image import get_threshold_mask


def get_vit_patch_features_and_clusters(
    image: str,
    model,
    create_mask: bool,
    overlap,
    sigma,
    window_size: list = [224, 224, -1],
    fill_holes: bool = True,
    leiden_resolution: float = 0.75,
    erosion_iterations: int = 1,
    calculate_cls_clusters: bool = False,
):
    # Switch channel to the -1th dimension
    # if image.shape[0] == 3:
    #         image = image.transpose(1, 2, 0)

    if create_mask:
        mask = get_threshold_mask(image, fill_holes=fill_holes)
    else:
        mask = None

    img_norm = histology_features.normalise_rgb(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform_no_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if overlap is not None:
        feature_blocks = histology_features.feature_map_overlap_blocks(
            img_norm,
            feature_extract_function=histology_features.transformer_features,
            mask=mask,
            window_size=window_size,
            overlap=overlap,
            model=model,
            transforms_to_apply=transform_no_norm,
        )
    else:
        feature_blocks = histology_features.feature_map_blocks2(
            img_norm,
            feature_extract_function=histology_features.transformer_features,
            mask=mask,
            window_size=window_size,
            # overlap=overlap,
            model=model,
            transforms_to_apply=transform_no_norm,
        )

    patch_features, cls_features = (
        histology_features.postprocess_dask_transformer_features(feature_blocks)
    )

    if mask is not None:
        resized_mask_patch = skimage.transform.resize(
            mask, output_shape=patch_features.shape[:-1], order=0
        )
        resized_mask_patch = scipy.ndimage.binary_erosion(
            resized_mask_patch, iterations=erosion_iterations
        )
        # patch_features = patch_features * resized_mask_patch[...,numpy.newaxis]
    else:
        resized_mask_patch = None

    patch_features_smoothed = skimage.filters.gaussian(
        patch_features, sigma=sigma, truncate=4, channel_axis=-1
    )

    patch_clusters, patch_cmap = histology_features.cluster_blocks(
        patch_features_smoothed,
        "leiden",
        mask=resized_mask_patch,
        resolution=leiden_resolution,
    )

    if calculate_cls_clusters:
        if mask is not None:
            resized_mask_cls = skimage.transform.resize(
                mask, output_shape=cls_features.shape[:-1], order=0
            )
            # resized_mask_patch = scipy.ndimage.binary_erosion(resized_mask_patch, iterations=erosion_iterations)

        cls_features_smoothed = skimage.filters.gaussian(
            cls_features, sigma=sigma, truncate=4, channel_axis=-1
        )
        cls_clusters, cls_cmap = histology_features.cluster_blocks(
            cls_features_smoothed,
            "leiden",
            mask=resized_mask_cls,
            resolution=leiden_resolution,
        )
        return (
            resized_mask_patch,
            mask,
            patch_features,
            patch_clusters,
            patch_cmap,
            cls_features,
            cls_clusters,
            cls_cmap,
            resized_mask_cls,
        )
    else:
        return (
            resized_mask_patch,
            mask,
            patch_features,
            patch_clusters,
            patch_cmap,
        )
