import pathlib
import typing

import anndata
import h5py
import numpy


# Adataped from SpatialData IO
def read_10x_h5(
    filename: typing.Union[str, pathlib.Path],
    genome: typing.Optional[str] = None,
    gex_only: bool = True,
) -> anndata.AnnData:
    """
    Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to a 10x hdf5 file.
    genome
        Filter expression to genes within this genome. For legacy 10x h5
        files, this must be provided if the data contains more than one genome.
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name.
    Stores the following information:

        - `~anndata.AnnData.X`: The data matrix is stored
        - `~anndata.AnnData.obs_names`: Cell names
        - `~anndata.AnnData.var_names`: Gene names
        - `['gene_ids']`: Gene IDs
        - `['feature_types']`: Feature types
    """
    start = print(f"reading {filename}")
    filename = pathlib.Path(filename) if isinstance(filename, str) else filename
    is_present = filename.is_file()
    if not is_present:
        print(f"... did not find original file {filename}")
    with h5py.File(str(filename), "r") as f:
        v3 = "/matrix" in f

    if v3:
        adata = _read_v3_10x_h5(filename, start=start)
        if genome:
            if genome not in adata.var["genome"].values:
                raise ValueError(
                    f"Could not find data corresponding to genome `{genome}` in `{filename}`. "
                    f'Available genomes are: {list(adata.var["genome"].unique())}.'
                )
            adata = adata[:, adata.var["genome"] == genome]
        if gex_only:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"]
        if adata.is_view:
            adata = adata.copy()
    else:
        raise ValueError("Versions older than V3 are not supported.")
    return adata


def _read_v3_10x_h5(
    filename: typing.Union[str, pathlib.Path],
    *,
    start: typing.Optional[typing.Any] = None,
) -> anndata.AnnData:
    """Read hdf5 file from Cell Ranger v3 or later versions."""
    with h5py.File(str(filename), "r") as f:
        try:
            dsets: dict[str, typing.Any] = {}
            _collect_datasets(dsets, f["matrix"])

            from scipy.sparse import csr_matrix

            M, N = dsets["shape"]
            data = dsets["data"]
            if dsets["data"].dtype == numpy.dtype("int32"):
                data = dsets["data"].view("float32")
                data[:] = dsets["data"]
            matrix = csr_matrix(
                (data, dsets["indices"], dsets["indptr"]),
                shape=(N, M),
            )
            adata = anndata.AnnData(
                matrix,
                obs={"obs_names": dsets["barcodes"].astype(str)},
                var={
                    "var_names": dsets["name"].astype(str),
                    "gene_ids": dsets["id"].astype(str),
                    "feature_types": dsets["feature_type"].astype(str),
                    "genome": dsets["genome"].astype(str),
                },
            )
            return adata
        except KeyError:
            raise Exception("File is missing one or more required datasets.")


def _collect_datasets(dsets: typing.Dict[str, typing.Any], group: h5py.Group) -> None:
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            dsets[k] = v[:]
        else:
            _collect_datasets(dsets, v)
