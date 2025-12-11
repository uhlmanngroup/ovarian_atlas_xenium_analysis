import spatialdata
from spatialdata.models import TableModel, Image2DModel, ShapesModel
import geopandas
import shapely
import numpy
import pandas
import anndata

from histology_features.embedding import assign_embedding_index, construct_embedding_table, compute_embedding_pca
from histology_features.utility import get_spatial_element

N_OBS = 5
N_VARS = 3
EMBEDDING_DIM = 64

def sample_anndata():
    """Create a simple anndata object with spatial coordinates."""
    
    obs = pandas.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)])

    var = pandas.DataFrame(index=[f"gene_{i}" for i in range(N_VARS)])

    X = numpy.random.rand(N_OBS, N_VARS)
    
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = numpy.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 1.0],
        [5.0, 5.0]
    ])
    
    # Add an annotation column
    adata.obs['annotation'] = pandas.Series([1, 1, 2, 2, 3], index=adata.obs.index)
    
    return adata

def sample_sdata():

    adata = sample_anndata()

    centroids = adata.obsm["spatial"].astype(int)

    img = numpy.zeros((EMBEDDING_DIM, 128, 128))

    # For the embedding image, assign the centroids an incrementing
    # value
    img[
        ...,
        centroids[:, 0],
        centroids[:, 1],
    ] = numpy.arange(len(centroids))

    cells = geopandas.GeoDataFrame(
        {
            "geometry": [shapely.Point(i, j).buffer(5) for i,j in centroids]
        }
    )

    images = {"embedding_image": Image2DModel.parse(img)}
    tables = {"table": TableModel.parse(adata)}
    shapes = {"cell_boundaries": ShapesModel.parse(cells)}
    
    data = {
        "images": images,
        "tables": tables,
        "shapes": shapes
    }
    sdata = spatialdata.SpatialData(
        **data
    )

    return sdata

class TestEmbedding:
    def test_embedding(self):
        sdata = sample_sdata()

        assign_embedding_index(
            sdata=sdata,
            embedding_image_key="embedding_image",
            patch_size=1,
            cell_shape_key="cell_boundaries",
            cell_shape_scale_factor=1,
        ) 

        construct_embedding_table(
            sdata=sdata,
            embedding_image_key="embedding_image",
            table_key="table",
            embedding_table_key="embeddings",
        )

        expected = numpy.zeros((N_OBS, EMBEDDING_DIM))
        expected[1, :] = 1
        expected[2, :] = 2
        expected[3, :] = 3
        expected[4, :] = 4

        observed = sdata.tables["embeddings"].X

        numpy.testing.assert_equal(expected, observed)

    def test_embedding_pca(self):
        sdata = sample_sdata()

        assign_embedding_index(
            sdata=sdata,
            embedding_image_key="embedding_image",
            patch_size=1,
            cell_shape_key="cell_boundaries",
            cell_shape_scale_factor=1,
        ) 

        construct_embedding_table(
            sdata=sdata,
            embedding_image_key="embedding_image",
            table_key="table",
            embedding_table_key="embeddings",
        )

        compute_embedding_pca(
            sdata,
            embedding_table_key="embeddings",
            n_components = 50,
        )

        assert sdata.tables["embeddings"].obsm["embeddings_pca"].shape == (5, 5)

