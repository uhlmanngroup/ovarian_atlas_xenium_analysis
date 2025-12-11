from histology_features.utility import get_spatial_element
import numpy
from spatialdata.models import TableModel
from sklearn.decomposition import PCA
import anndata
import shapely
import geopandas
import dask.array
from zarr.core import Array
from xarray.core.dataarray import DataArray

def assign_embedding_index(
    sdata,
    embedding_image_key: str,
    patch_size: int,
    cell_shape_key: str | None = None,
    cell_shape_scale_factor: float = None,
):
    embedding_image = get_spatial_element(
        sdata.images,
        embedding_image_key,
        as_spatial_image=True
    )

    if cell_shape_key:
        centroids = get_spatial_element(
            sdata.shapes,
            cell_shape_key,
        ).centroid
    else:
        centroids = sdata.tables["table"].obsm["spatial"]

        # XY to YX
        centroids = centroids[:, ::-1]

        points = [shapely.geometry.Point(xy) for xy in centroids]

        centroids = geopandas.GeoDataFrame(
            {"geometry": points}
        )
    
    if cell_shape_scale_factor is not None:
        centroids = centroids.scale(
            cell_shape_scale_factor,
            cell_shape_scale_factor,
            origin=(0, 0),
        )

    centroids = [(point.x, point.y) for point in centroids.geometry]
    
    centroids = numpy.divide(centroids, patch_size)

    centroids = numpy.round(centroids, 0)

    sdata.tables["table"].obs["embedding_idx_y"] = centroids[:, 0]
    sdata.tables["table"].obs["embedding_idx_x"] = centroids[:, 1]

def construct_embedding_table(
    sdata,
    embedding_image_key: str,
    table_key: str = "table",
    embedding_table_key: str = "histology_embeddings",
    nan_threshold: float = 0.05,
):
    """Create a new adata table for a SpatialData object
    containing the histology_embeddings per cell"""

    original_adata = sdata.tables[table_key]

    idx_y, idx_x = original_adata.obs["embedding_idx_y"].to_numpy(), original_adata.obs["embedding_idx_x"].to_numpy()

    embedding_image = get_spatial_element(
        sdata.images, 
        embedding_image_key, 
        as_spatial_image=True
    ).data

    embeddings = embedding_image.compute()

    # Transpose to n_obs × n_vars
    embeddings = embeddings[
        ...,
        idx_y.astype(int),
        idx_x.astype(int),
    ].T

    nan_embeddings = numpy.sum(numpy.isnan(embeddings).any(axis=1))
    perc_nan = (nan_embeddings / embeddings.shape[0])
    if perc_nan >= nan_threshold:
        raise ValueError(f"The percentage of NaN embeddings ({perc_nan}) is above the nan_threshold {nan_threshold}. Check that the patch_size passed to assign_embedding_index is correct.")
    
    embedding_adata = anndata.AnnData(embeddings)

    embedding_adata.obs.index = original_adata.obs.index

    sdata.tables[embedding_table_key] = TableModel.parse(
        embedding_adata
    )

def add_roi_embeddings(
    sdata,
    embeddings,
    table_key: str = "table",
    obsm_key: str = "histology_embeddings",
    feature_names: list[str] = None,
):
    """
    Add ROI embeddings to the .obsm of an existing table in a SpatialData object.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to add the embeddings to
    embeddings : Array, DataArray, or array-like
        The embeddings to add
    table_key : str, default "table"
        Key of the table in sdata
    obsm_key : str, default "histology_embeddings"
        Key under which the embeddings will be stored in .obsm
    
    Returns
    -------
    None
        Modifies sdata in place
    """
    original_adata = sdata.tables[table_key]

    original_n_obs = original_adata.shape[0]
    roi_obs = embeddings.shape[0]

    if original_n_obs != roi_obs:
        raise ValueError(
            f"Number of observations is mis-matched. "
            f"Expected {original_n_obs}, got {roi_obs}."
        )

    # Handle array type
    if isinstance(embeddings, Array):
        embeddings = dask.array.from_zarr(embeddings, chunks="auto")
    elif isinstance(embeddings, DataArray):
        embeddings = embeddings.data

    # Add embeddings into obsm
    original_adata.obsm[obsm_key] = embeddings

    if feature_names is not None:
        original_adata.uns[f"{obsm_key}_feature_names"] = feature_names


def compute_embedding_pca(
    sdata,
    embedding_table_key,
    embedding__pca_table_key,
    n_components: int = 50,
):
    
    adata = sdata.tables[embedding_table_key]
    X = adata.X
    n_obs = X.shape[0]

    n_components = n_obs if n_obs < n_components else n_components

    pca = PCA(n_components=n_components)
    pca.fit(X)

    adata.obsm[embedding__pca_table_key] = pca.transform(X)