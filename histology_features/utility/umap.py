import matplotlib
import numpy
import pandas
import seaborn
import umap


def umap_embedding(array: numpy.array) -> numpy.array:
    """Create a UMAP embedding from features"""
    reducer = umap.UMAP()

    embedding = reducer.fit_transform(array)

    return embedding


def plot_umap(feature_df: pandas.DataFrame(), row_id_col: str):
    feature_values = feature_df.drop(labels=row_id_col, axis=1).values

    embedding = umap_embedding(feature_values)

    df = pandas.DataFrame(embedding, columns=["umap_0", "umap_1"])

    df = pandas.concat([feature_df[row_id_col], df], axis=1)

    fig, ax = matplotlib.pyplot.subplots()
    seaborn.scatterplot(
        data=df, x="umap_0", y="umap_1", hue="category", palette="Set2", ax=ax
    )
    return fig
