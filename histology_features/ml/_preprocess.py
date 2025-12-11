import numpy
from scipy.sparse import issparse

def preprocess_adata(
    train_adata,
    target_adata,
    target_obs_column_name: str,
    train_embedding_key: str = None, 
    obs_to_drop: str | list[str] = None,
    merge_obs: bool = False,
    n_components: int | None = None,
    ):
    """
    merge_obs: If multiple cells are in the same XY patch,
    merge their observations into one. 
    """

    assert train_adata.n_obs == target_adata.n_obs, f"Input anndata objects must have the same number of observations. Got {train_adata.n_obs} and {target_adata.n_obs}"

    # Drop NaN values from train embeddings
    train_array = get_anndata_array(train_adata, train_embedding_key)
    if issparse(train_array):
        # Select sparse array data
        nan_mask = ~numpy.isnan(train_array.toarray()).any(axis=1)
    else:
        nan_mask = ~numpy.isnan(train_array).any(axis=1)

    train_adata = train_adata[nan_mask, :]
    target_adata = target_adata[nan_mask, :]

    if obs_to_drop is not None:
        if isinstance(obs_to_drop, str):
            obs_to_drop = [obs_to_drop]

        # Drop oberservations to not be used in training
        drop_mask = ~target_adata.obs[target_obs_column_name].isin(obs_to_drop)
        train_adata = train_adata[drop_mask, :]
        target_adata = target_adata[drop_mask, :]

    if merge_obs:
        grouped = (
            target_adata.obs.groupby(['embedding_idx_y', 'embedding_idx_x'])[target_obs_column_name]
            .unique()
            .apply(lambda x: frozenset(x))
            .reset_index()
        )
        new_target_obs_column_name = "all_"+target_obs_column_name
        grouped.rename(columns={target_obs_column_name: new_target_obs_column_name}, inplace=True)
        target_obs_column_name = new_target_obs_column_name
        target_adata.obs = target_adata.obs.merge(grouped, on=['embedding_idx_y', 'embedding_idx_x'], how='left')

    x = get_anndata_array(train_adata, train_embedding_key)
    y = target_adata.obs[target_obs_column_name].to_numpy()

    assert x.shape[0] == len(y), f"Number of training samples ({x.shape[0]}) != training labels ({len(y)})"

    if n_components is not None:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x)

    return x, y

def get_anndata_array(adata, key=None):
    if key.casefold() == "x" or key is None:
        if issparse(adata.X):
            return adata.X.toarray()
        else:
            return adata.X
    elif key in adata.obsm:
        if issparse(adata.obsm[key]):
            return adata.obsm[key].toarray()
        else:
            return adata.obsm[key]
    else:
        raise KeyError(f"Key '{key}' not found in adata.obsm.")