import typing

import anndata
import numpy
import scanpy
from harmonypy import run_harmony


def batch_correct_blocks(
    feature_blocks_dict: typing.Dict[int, numpy.ndarray],
    batch_correction_method: typing.Literal["scvi", "harmony", "scanorama"],
    mask_blocks_dict: typing.Dict[int, numpy.ndarray] = None,
    batch_key: str = "batch",
    masked_value=numpy.nan,
    **kwargs,
):
    """Perform batch correction on sequential feature block images.

    feature_blocks_dict: a dictionary with batch_id as key and feature blocks
    as value.
    """

    adata = None

    # Dictionary to store batch key and the indices to mask
    batch_foreground_idx = {}

    for (batch_id, feature_blocks), (_, mask) in (
        zip(feature_blocks_dict.items(), mask_blocks_dict.items())
        if mask_blocks_dict is not None
        else zip(
            feature_blocks_dict.items(),
            dict.fromkeys(set(range(len(feature_blocks_dict)))).items(),
        )
    ):
        # Target shape is (n_blocks, features)
        target_shape = (
            feature_blocks.shape[0] * feature_blocks.shape[1],
            feature_blocks.shape[-1],
        )

        # Reshape to array of (num_blocks, num_features_per_block)
        reshaped_features = numpy.reshape(feature_blocks, target_shape)

        if mask is not None:
            reshaped_mask = numpy.reshape(
                mask, (feature_blocks.shape[0] * feature_blocks.shape[1], 1)
            )

            # Define indices that are not masked
            idx_to_cluster = numpy.where(reshaped_mask)[0]
        else:
            # No mask provided, so all blocks will be clustered
            idx_to_cluster = numpy.arange(0, reshaped_features.shape[0])

        batch_foreground_idx[batch_id] = idx_to_cluster

        # Use an AnnData object as the central feature store
        # for batch correction. Since we're using single cell batch
        # correction techniques, there's wide support for AnnData
        loop_adata = anndata.AnnData(reshaped_features)[idx_to_cluster]
        # Set the batch identifier (the dictionary key)
        loop_adata.obs[batch_key] = batch_id
        if adata is None:
            adata = loop_adata
        else:
            adata = anndata.concat([adata, loop_adata])

    # Convert batch ID to a string. Harmony requires this for
    # some reason
    adata.obs[batch_key] = adata.obs[batch_key].apply(str)

    # Scale values so -ve is 0. Batch correction
    # methods for single-cell data works with counts
    # so expects non-negative values
    min_value = adata.X.min()
    adata.X -= min_value

    if batch_correction_method.casefold() == "scvi":
        features = scvi_batch_correct(adata, batch_key=batch_key, **kwargs)
    elif batch_correction_method.casefold() == "harmony":
        features = harmony_batch_correct(adata, batch_key=batch_key, **kwargs)
    elif batch_correction_method.casefold() == "scanorama":
        features = scanorama_batch_correct(adata, batch_key=batch_key, **kwargs)
    elif batch_correction_method.casefold() == "pass":
        features = adata.X
    else:
        raise ValueError

    corrected_adata = anndata.AnnData(features)
    # Add the batch back in
    # This is so convoluted...
    corrected_adata.obs[batch_key] = adata.obs[batch_key].values

    output_data = {}
    for batch_id, feature_blocks in feature_blocks_dict.items():
        # Target shape is (n_blocks, features)
        # Recover the original shape again
        target_shape = (
            feature_blocks.shape[0] * feature_blocks.shape[1],
            feature_blocks.shape[-1],
        )
        # Construct a output_feature store, which includes both
        # masked and batch corrected features
        # output_features with size (num_blocks, num_features)
        output_features = numpy.zeros(target_shape)

        # For this batch, get the indices that refer to foreground
        # blocks
        idx_to_add = batch_foreground_idx[batch_id]

        # Batch ID is stored as a string
        assert (
            output_features[idx_to_add].shape
            == corrected_adata[corrected_adata.obs[batch_key] == str(batch_id)].X.shape
        )

        # For each batch, add the batch corrected values
        # ie. those that were not masked
        # Indices are stored independently for each batch, so add accordingly
        output_features[idx_to_add] = corrected_adata[
            corrected_adata.obs[batch_key] == str(batch_id)
        ].X

        # Find the masked indices and set to the masked_value
        if mask is not None:
            inverted_idx = numpy.setdiff1d(range(output_features.shape[0]), idx_to_add)
            output_features[inverted_idx] = masked_value

        # Reshape the "flat" features back into their original feature_block shape
        output_features = numpy.reshape(output_features, feature_blocks.shape)

        output_data[batch_id] = output_features

    return output_data


def scvi_batch_correct(adata, batch_key="batch", n_layers=2, n_latent=30):

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30)

    vae.train()

    features = vae.get_latent_representation()

    return features


def harmony_batch_correct(adata, batch_key="batch", max_iter_harmony=20):
    harmony_output = run_harmony(
        adata.X,
        adata.obs,
        [batch_key],
        max_iter_harmony=max_iter_harmony,
    )
    features = harmony_output.Z_corr.T

    return features


def scanorama_batch_correct(adata, batch_key="batch"):
    # Sort batch labels based on batch key
    idx = adata.obs[batch_key].argsort().values
    adata = adata[idx, :]
    # arpack
    n_latent = min(adata.shape) - 1
    scanpy.tl.pca(adata, n_comps=n_latent)

    scanpy.external.pp.scanorama_integrate(
        adata, batch_key, adjusted_basis="X_scanorama"
    )
    features = adata.obsm["X_scanorama"]

    return features
