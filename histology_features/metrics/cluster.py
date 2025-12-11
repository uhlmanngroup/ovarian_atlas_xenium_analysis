import itertools

import numpy
import sklearn


def compare_different_cluster_blocks(
    cluster_blocks_gt: numpy.ndarray, cluster_blocks_pred: numpy.ndarray
) -> numpy.ndarray:
    assert (
        cluster_blocks_gt.shape == cluster_blocks_pred.shape
    ), "Arrays must be the same shape."

    # Remove nan
    cluster_blocks_gt = cluster_blocks_gt[~numpy.isnan(cluster_blocks_gt)]
    cluster_blocks_pred = cluster_blocks_pred[~numpy.isnan(cluster_blocks_pred)]

    # Ensure the same pixels are masked (and are thus NaN)
    assert (
        cluster_blocks_gt.shape == cluster_blocks_pred.shape
    ), "Unequal number of NaNs."

    gt_cluster_ids = numpy.unique(cluster_blocks_gt).astype(int)
    pred_cluster_ids = numpy.unique(cluster_blocks_pred).astype(int)

    # Get all cluster combinations
    cluster_id_combinations = list(itertools.product(gt_cluster_ids, pred_cluster_ids))

    f1_score_matrix = numpy.zeros((len(gt_cluster_ids), len(pred_cluster_ids)))

    for clstr_cmb in cluster_id_combinations:
        gt = numpy.where(cluster_blocks_gt == clstr_cmb[0], True, False)
        pred = numpy.where(cluster_blocks_pred == clstr_cmb[1], True, False)

        f1 = sklearn.metrics.f1_score(gt, pred)

        f1_score_matrix[clstr_cmb[0], clstr_cmb[1]] = f1

    return f1_score_matrix
