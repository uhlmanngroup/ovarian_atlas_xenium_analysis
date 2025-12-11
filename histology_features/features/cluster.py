import typing

import anndata
import matplotlib.pyplot as plt
import numpy
import scanpy
from sklearn.cluster import HDBSCAN, KMeans

from ..utility.colormap import make_cmap


def cluster_blocks(
    feature_blocks: numpy.array,
    cluster_method: typing.Literal["kmeans", "hdbscan", "leiden"],
    mask: numpy.array = None,
    masked_value: float = numpy.nan,
    return_adata: bool = False,
    volumetric: bool = False,
    **kwargs,
):
    """
    cluster the output of feature_map_blocks, which is
    an array with shape (x_dims, y_dims, feature_vector)
    """

    if not volumetric and feature_blocks.ndim == 4:
        raise ValueError(
            f"Got a feature block image with ndim {feature_blocks.ndim} suggesting a 3D feature map but volumetric is {volumetric}."
        )

    if volumetric and not feature_blocks.ndim == 4:
        raise ValueError(
            f"Got volumetric = {volumetric} but feature blocks is not 3D and instead has {feature_blocks.ndim} dims (expected 4)"
        )

    if volumetric:
        # If volumetric, the 0th dimension contains the z-planes
        target_shape = (
            feature_blocks.shape[0] * feature_blocks.shape[1] * feature_blocks.shape[2],
            feature_blocks.shape[-1],
        )
    else:
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
        if volumetric:
            reshaped_mask = numpy.reshape(
                mask,
                (
                    feature_blocks.shape[0]
                    * feature_blocks.shape[1]
                    * feature_blocks.shape[2],
                    1,
                ),
            )
        else:
            reshaped_mask = numpy.reshape(
                mask, (feature_blocks.shape[0] * feature_blocks.shape[1], 1)
            )
        # Define indices that are not masked
        idx_to_cluster = numpy.where(reshaped_mask)[0]
    else:
        # No mask provided, so all blocks will be clustered
        idx_to_cluster = numpy.arange(0, reshaped_features.shape[0])

    if cluster_method.casefold() == "kmeans":
        cluster_id = KMeans(**kwargs).fit_predict(reshaped_features[idx_to_cluster])
    elif cluster_method.casefold() == "hdbscan":
        cluster_id = HDBSCAN(**kwargs).fit_predict(reshaped_features[idx_to_cluster])
    elif cluster_method.casefold() == "leiden":
        adata = anndata.AnnData(reshaped_features)[idx_to_cluster]
        try:
            # Use GPU, if available
            import rapids_singlecell

            print("Using GPU-accelerated clustering")
            rapids_singlecell.pp.neighbors(adata, use_rep="X")
            rapids_singlecell.tl.leiden(adata, **kwargs)
            cluster_id = adata.obs.leiden.to_numpy().astype(int)
            rapids_singlecell.tl.umap(adata)
        except:
            print("No GPU detected. Using CPU clustering")
            # We use pynndescent to find KNN since Scanpy's
            # defaults (which also uses automatic brute forcing for finding KNN
            # in small datasets for speed) to prevent indexing errors.
            scanpy.pp.neighbors(adata, use_rep="X", transformer="pynndescent")
            scanpy.tl.leiden(adata, **kwargs)
            cluster_id = adata.obs.leiden.to_numpy().astype(int)
            # Create and plot UMAP. This generates the unstructured "leiden_colors"
            # in adata, which can be used to generate a cmap. This allows for UMAP
            # and feature block images to have spatially mapped clusters for
            # visualisation Downside: this returns an inline UMAP plot, though this
            # is sort of handy to always have.
            scanpy.tl.umap(adata)
        if not return_adata:
            num_clusters = adata.obs.leiden.unique().categories.shape[0]
            if num_clusters > 100:
                raise Warning(
                    f"Number of leiden clusters ({num_clusters}) too high to plot. Consider reducing leiden resolution."
                )
            with plt.rc_context({"figure.figsize": (5, 5), "figure.dpi": (300)}):
                scanpy.pl.umap(adata, color=["leiden"])

            cmap = make_cmap(adata.uns["leiden_colors"])

    else:
        raise NotImplementedError

    output_features = numpy.zeros((target_shape[0], 1))
    output_features[idx_to_cluster] = cluster_id[..., numpy.newaxis]

    if mask is not None:
        inverted_idx = numpy.setdiff1d(range(output_features.shape[0]), idx_to_cluster)
        output_features[inverted_idx] = masked_value

    if volumetric:
        output_features = numpy.reshape(
            output_features,
            (
                feature_blocks.shape[0],
                feature_blocks.shape[1],
                feature_blocks.shape[2],
                1,
            ),
        )
    else:
        output_features = numpy.reshape(
            output_features, (feature_blocks.shape[0], feature_blocks.shape[1], 1)
        )

    if return_adata:
        return output_features, adata
    elif cluster_method.casefold() == "leiden" and return_adata == False:
        return output_features, cmap
    else:
        return output_features


def cluster_batch_blocks(
    feature_blocks_dict: dict,
    cluster_method: typing.Literal["kmeans", "hdbscan", "leiden"],
    masked_value=None,
    **kwargs,
):
    # Reshape dict values into a single array
    feature_blocks = numpy.array([i for i in feature_blocks_dict.values()])

    if masked_value is not None:
        if numpy.isnan(masked_value):
            # Get the mask from where blocks are nan'd
            mask = numpy.where(numpy.isnan(feature_blocks), False, True)
            mask = mask[..., 0]
        elif masked_value == 0:
            mask = numpy.where(feature_blocks == masked_value, False, True)
            mask = mask[..., 0]
        else:
            raise ValueError
    else:
        raise ValueError

    return cluster_blocks(
        feature_blocks, cluster_method, mask=mask, volumetric=True, **kwargs
    )


def get_cluster_mask_consensus(
    clusters1: numpy.ndarray, clusters2: numpy.ndarray
) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    """In histology_features, we set background pixels to
    NaN to prevent these pixels from influencing clustering.

    However, the mask that is used to define these background
    pixels is often resized to match the resolution of the
    feature block image.

    As a result, the resized mask may have resulting NaN values
    in one image where they are not in the other. This happens at
    the **border** of the foreground object.

    This function compares the NaN values of two cluster images
    and finds a better mask between the two: where there is a consensus
    of NaN values. If the NaN occurs in only one cluster
    image, the other image also sets the same element as NaN.

    Alternative solution to this problem: a binary mask dilation and
    erosion earlier in the pipeline.
    """

    assert clusters1.shape == clusters2.shape, "Cluster images should be equal"

    # If NaN is in both images, that element is True. Otherwise, False.
    shared_mask = numpy.logical_or(numpy.isnan(clusters1), numpy.isnan(clusters2))

    # Set all individual and join NaNs to NaN in both images
    clusters1_fixed = numpy.where(~shared_mask, clusters1, numpy.nan)
    clusters2_fixed = numpy.where(~shared_mask, clusters2, numpy.nan)

    assert (
        numpy.isnan(clusters1_fixed).shape == numpy.isnan(clusters2_fixed).shape
    ), "NaN's not equal"

    return clusters1_fixed, clusters2_fixed
