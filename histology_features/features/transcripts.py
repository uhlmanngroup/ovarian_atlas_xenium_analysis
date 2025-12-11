import typing

import polars

# from histology_features.features.array import rescale_array
import scanpy
from sklearn.preprocessing import LabelEncoder

from histology_features.features.read_10x_h5 import read_10x_h5
from histology_features.spec.xenium import Xenium


def scale_and_round_xenium_transcripts(
    df_path: str,
    scale_level: int,
    level_shape: typing.Tuple,
    coordinate_space: typing.Literal["pixel", "micron"],
    x_column_name: str = "x_location",
    y_column_name: str = "y_location",
    z_column_name: str = "feature_name",
    quality_measure_column_name: str = "qv",
    overlapping_transcript_aggregation_method: typing.Literal[
        "mean",
        "max",
        "sum",
    ] = "mean",
    qv_threshold: int = 20,
    gene_subset: typing.List[str] = None,
    keep_highly_variable: bool = False,
    cell_matrix_h5_path: str = None,
    sample: int = None,
    sample_seed: int = 42,
):
    """
    level_shape: The shape (X, Y) of the equivalent image eg. the H&E or 
        DAPI images. level_shape is important in ensuring no transcripts
        are rendered during cropping outside of the image shape range.
    coordinate_space: The target coordinate space. For downstream applications
        with images, select pixel (since images are in pixel space). To 
        leave the transcript coordinates unchanged (but still scaled), select 
        micron.
    qv_threshold: Value of 20 was determined based upon what 10X recommend
        for removing low quality transcript from Xenium
    """
    # Read parquet file with Polars
    df = polars.read_parquet(
        df_path,
        columns=[
            x_column_name,
            y_column_name,
            z_column_name,
            quality_measure_column_name,
        ],
    )

    # Set dtypes for low memory usage
    df = df.with_columns(
        [
            polars.col(x_column_name).cast(polars.Float32),
            polars.col(y_column_name).cast(polars.Float32),
            polars.col(quality_measure_column_name).cast(polars.Float32),
            polars.col(z_column_name).cast(polars.Categorical),
        ]
    )

    if sample is not None:
        df = df.sample(fraction=sample, seed=sample_seed)

    if gene_subset is not None:
        df = df.filter(
            polars.col(z_column_name)
            .cast(polars.Utf8)
            .str.contains("|".join(gene_subset))
        )

    # Remove bad transcripts
    # eg. control probes
    bad_transcripts = Xenium.bad_transcripts()
    df = df.filter(
        ~polars.col(z_column_name)
        .cast(polars.Utf8)
        .str.contains("|".join(bad_transcripts))
    )

    if keep_highly_variable:
        if cell_matrix_h5_path is None:
            raise ValueError(
                "Must provide cell_matrix_h5_path if requesting keep_highly_variable."
            )
        adata = read_10x_h5(cell_matrix_h5_path)
        scanpy.pp.log1p(adata)
        hvg_df = scanpy.pp.highly_variable_genes(adata, inplace=False)
        non_hvg = hvg_df[hvg_df["highly_variable"] == False].index.values
        print(f"Removing {len(non_hvg)} non-highly variable genes")
        df = df.filter(
            ~polars.col(z_column_name).cast(polars.Utf8).str.contains("|".join(non_hvg))
        )

    if qv_threshold is not None:
        df = df.filter(polars.col(quality_measure_column_name) >= qv_threshold)

    # Create pixel_value column that will store a value of 1 for each transript
    # This represents that a given pixel has a gene present.
    # Later, this information can be combined with adjacent pixels (eg. mean or sum)
    df = df.with_columns([polars.lit(1).alias("pixel_value")])

    if coordinate_space.casefold() == "pixel":
        df = df.with_columns(
            [
                (
                    polars.col(x_column_name)
                    / Xenium.pixel_size.value
                    / (2**scale_level)
                ).alias(x_column_name),
                (
                    polars.col(y_column_name)
                    / Xenium.pixel_size.value
                    / (2**scale_level)
                ).alias(y_column_name),
            ]
        )
    elif coordinate_space.casefold() == "micron":
        df = df.with_columns(
            [
                (polars.col(x_column_name) / (2**scale_level)).alias(x_column_name),
                (polars.col(y_column_name) / (2**scale_level)).alias(y_column_name),
            ]
        )
    else:
        raise ValueError(f"coordinate_space {coordinate_space} not recognised.")

    # Replace division by 0 with 0
    df = df.with_columns(
        [
            polars.when(polars.col(x_column_name).is_infinite())
            .then(0)
            .otherwise(polars.col(x_column_name))
            .alias(x_column_name),
            polars.when(polars.col(y_column_name).is_infinite())
            .then(0)
            .otherwise(polars.col(y_column_name))
            .alias(y_column_name),
        ]
    )

    # Round XY and qv to the nearest int
    df = df.with_columns(
        [
            polars.col(x_column_name).round(0).cast(polars.Int32).alias(x_column_name),
            polars.col(y_column_name).round(0).cast(polars.Int32).alias(y_column_name),
        ]
    )

    # Clip values to fall within the range
    df = df.with_columns(
        [
            polars.col(y_column_name).clip(0, level_shape[0] - 1).alias(y_column_name),
            polars.col(x_column_name).clip(0, level_shape[1] - 1).alias(x_column_name),
        ]
    )

    def label_encoder(series: polars.Series) -> polars.Series:
        le = LabelEncoder()
        labels = le.fit_transform(series.to_numpy())
        return polars.Series(labels)

    # Encode labels to get z-dimensions from string feature names
    df = df.with_columns(
        polars.col(z_column_name)
        .map_batches(label_encoder)
        .alias(f"{z_column_name}_idx")
    )

    # Aggregate overlapping transcripts
    # TODO: mean and max will return the same if "pixel_value" is all 1s. Need to consider alterantive aggregation methods
    if overlapping_transcript_aggregation_method.casefold() == "mean":
        df = df.group_by(
            [x_column_name, y_column_name, z_column_name, f"{z_column_name}_idx"]
        ).agg(polars.mean("pixel_value").alias("pixel_value"))
    elif overlapping_transcript_aggregation_method.casefold() == "max":
        df = df.group_by(
            [x_column_name, y_column_name, z_column_name, f"{z_column_name}_idx"]
        ).agg(polars.max("pixel_value").alias("pixel_value"))
    elif overlapping_transcript_aggregation_method.casefold() == "sum":
        df = df.group_by(
            [x_column_name, y_column_name, z_column_name, f"{z_column_name}_idx"]
        ).agg(polars.sum("pixel_value").alias("pixel_value"))
    else:
        raise ValueError(
            f"overlapping_transcript_aggregation_method: {overlapping_transcript_aggregation_method} not recognised"
        )

    return df
