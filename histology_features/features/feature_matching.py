import numpy
import sklearn.neighbors


def match_features(
    fit_features: numpy.ndarray,
    query_features: numpy.ndarray,
    idx_to_keep: numpy.ndarray = None,
    n_neighbors: int = 1,
    remove_zero_distance: bool = True,
) -> numpy.ndarray:
    """For a two feature maps, find the 1 KNN for each feature vector in the fit image.

    Returned fit to query indices will be sorted based on the KNN distances. Thus, top indices
    will be the 'most simimar'."""
    # Instantiate KNN
    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    # Fit to the input features
    knn.fit(fit_features)
    # For the query features, find the nearest neighbour of the fit features
    distances, match2to1 = knn.kneighbors(query_features)

    # Create an array with 3 columns
    # 0th: the distances
    # 1st: the index in the fit features image
    # 2nd: the index for the query features

    # Here we use numpy.arange(match2to1.shape[0]) to get the query image indices.
    # This is because match2to1 contains the indices of KNNs of the fit image
    # The shape of match2to1 is the same shape as the query image.
    # For example, the 0th row of match2to1 has a value of 2. This means the 0th
    # feature vector in the query features is most similar to the 2nd row index
    # in the fit features.
    query_indices = numpy.stack(
        [numpy.arange(match2to1.shape[0])] * n_neighbors, axis=1
    )

    # Stack the arrays. The output shape will be as follows:
    # (n_features, n_neighbours, 3)
    # Thus,
    combined = numpy.stack([distances, match2to1, query_indices], axis=2)

    # Remove background features
    combined = combined[idx_to_keep]

    if remove_zero_distance:
        # I couldn't work out how to do this without
        # iteration :'[
        output = []
        for i in range(combined.shape[0]):
            arr = combined[i, combined[i][:, 0] > 0]
            if arr.size != 0:
                output.append(arr)
        combined = numpy.array(output)

    output = []
    for i in range(combined.shape[0]):
        arr = combined[i, combined[i][:, 0] != numpy.nan]
        if arr.size != 0:
            output.append(arr)
    combined = numpy.array(output)

    return combined

    # For each feature vector query, sort the output array
    # based on distance. This, combined[0] represents
    # the "nearest" feature vectors
    combined = combined[:, combined[:, :, 0].mean(0).argsort()]

    return combined


def visualise_matched_features(
    knn_array: numpy.ndarray,
    fit_image: numpy.ndarray,
    query_image: numpy.ndarray,
    number_matches_to_plot: int = 5,
    number_of_neighbours_per_patch: int = 1,
    fit_image_cmap=None,
    query_image_cmap=None,
    figsize=(10, 10),
    top_k: int = 20,
) -> None:
    """
    Plot lines between a fit and query image to demonstrate the region
    of the nearest neighbour feature.

    knn_array is returned by match_features and has the following (**sorted**)
    column structure:
    0th: the distances
    1st: the index in the fit features image
    2nd: the index for the query features
    """

    assert (
        knn_array.shape[1] >= number_of_neighbours_per_patch
    ), "Can't plot more neighbours than have been measured. Either lower number_of_neighbours_per_patch or find more KNNs"

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(fit_image, cmap=fit_image_cmap)
    ax[1].imshow(query_image, cmap=query_image_cmap)

    ax[0].set_title("Fit image")
    ax[1].set_title("Query image")

    for i in range(number_matches_to_plot):
        # Randomy sample the number of top matches
        plot_idx = numpy.random.randint(0, top_k)

        # Get the array indices for the query and fit
        # plot_idx refers to the feature_vector to plot
        # In knn_array, the
        for i in range(number_of_neighbours_per_patch):
            arr1_ind = int(knn_array[plot_idx][i][1])
            arr2_ind = int(knn_array[plot_idx][i][2])

            # Translate the array index to an actual index in the image
            arr1_ind = numpy.unravel_index(arr1_ind, fit_image.shape[:2])
            arr2_ind = numpy.unravel_index(arr2_ind, query_image.shape[:2])

            # Flip the coords sicne matplotlib scatter
            # uses (col, row) coords
            arr1_ind = numpy.flip(arr1_ind)
            arr2_ind = numpy.flip(arr2_ind)

            # Create the connection "line"
            con = ConnectionPatch(
                xyA=arr1_ind,
                xyB=arr2_ind,
                coordsA="data",
                coordsB="data",
                axesA=ax[0],
                axesB=ax[1],
                color=numpy.random.rand(
                    3,
                ),
            )
            ax[1].add_artist(con)

    [_ax.axis("off") for _ax in ax.flatten()]
