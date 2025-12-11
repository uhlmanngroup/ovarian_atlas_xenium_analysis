import typer
import typing
import pathlib
import tempfile
import subprocess

# from ._align import align
from histology_features.spec import Constants
import tomllib
import os
import ast

from ._search import search
from ._submit import submit_slurm_job
from ..annotation import merge_annotations as _merge_annotations
from histology_features.annotation import relate_annotations
from histology_features.utility import relate_keys

from InquirerPy import inquirer

app = typer.Typer()

SLURM_COMMANDS = [
    "concatenate",
    "submit_registration",
]

SLURM_TIME = "08:00:00"
SLURM_CPU_PER_TASK = 24
SLURM_MEMORY = "256G"

@app.command()
def slurm_submit(
    config_file: str
):

    selected_command = inquirer.select(
        message="Select a command to submit as a SLURM job:",
        choices=SLURM_COMMANDS,
    ).execute()

    container_path = None

    container_path = (
        container_path
        if not inquirer.confirm(f"Use singularity?", default=False).execute()
        else inquirer.text("Enter path to container: ").execute()
    )

    submit_slurm_job(
        selected_command=selected_command,
        container_path=container_path,
        config_file=config_file,
    )

@app.command()
def concatenate(
    config_file: str
):
    from ..io import spatialdata_concat

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    spatialdata_concat(
        sdata_path_list=config.get("concatenate").get("file_paths"),
        save_path=config.get("concatenate").get("save_path"),
        table_only = config.get("concatenate").get("table_only", False),
        drop_cols = config.get("concatenate").get("drop_cols", None),
        output_obs_csv = config.get("concatenate").get("output_obs_csv", False),
    )

@app.command()
def submit_registration(
    config: str
):
    from histology_features.io.histology import ImageAlign

    config = pathlib.Path(config)

    if config.is_dir():
        config_path = inquirer.fuzzy("Select a config file", choices=search(config, ".toml")).execute()
    elif config.suffix == ".toml":
        config_path = config
    else:
        raise ValueError

    config = tomllib.load(open(config_path, "rb"))

    source_images = config.get("image", None)

    assert source_images is not None, "[[image]] must be provided in the config toml. Got None."

    # TODO: Move this to validate_input
    target_image = None
    for img in source_images:
        if img.get("target_image", None) is not None:
            # Check if target image has already been set
            assert target_image == None, "Multiple target images found. Can only have 1 target image"
            target_image = img

    save_path = config["alignment"].get("save_path", None)
    transformation = config["alignment"].get("transformation", None)

    # save_path = (
    #     save_path
    #     if inquirer.confirm(f"Confirm save path for output SpatialData: {save_path}", default=True).execute()
    #     else inquirer.text("Enter the desired save path:").execute()
    # )

    save_full_slide = config["alignment"].get("save_full_slide", False)

    save_path = pathlib.Path(save_path)
    if not save_path.suffix == ".zarr":
        save_path = save_path.with_name(save_path.name + ".zarr")

    alignment_kwargs = config["alignment"].get("alignment_kwargs", {})

    aligner = ImageAlign(
        source_images=source_images,
        alignment_method = "valis",
        save_full_slide = save_full_slide,
        alignment_kwargs = alignment_kwargs,
        save_path=save_path,
        transformation=transformation,
        overwrite=config["alignment"].get("overwrite", False),
        dimensions=config["alignment"].get("dimensions", {"c": 0, "x": 1, "y": 2}),
    )

    sdata = aligner.process()

@app.command()
def add_batch_column(config_file: str):
    import spatialdata
    config = tomllib.load(open(config_file, "rb"))

    sdata_path = config.get("spatialdata_path")
    
    batch_id = config.get("batch_id")

    save_path = config.get("save_path")

    assert batch_id is not None, "Must provide a batch_id"

    sdata = spatialdata.read_zarr(sdata_path)

    sdata["table"].obs["batch"] = batch_id

    sdata.write(
        save_path, 
        overwrite=config.get("overwrite", True)
    )

@app.command()
def add_embedding(config_file: str):
    import spatialdata
    from histology_features.embedding import add_roi_embeddings
    import zarr

    config = tomllib.load(open(config_file, "rb"))

    sdata_path = config.get("spatialdata_path")

    embedding_path = config.get("embedding_path")

    save_path = config.get("save_path")

    sdata = spatialdata.read_zarr(sdata_path)

    embeddings = zarr.open(embedding_path)

    add_roi_embeddings(sdata, embeddings)

    sdata.write(
        save_path, 
        overwrite=config.get("overwrite", True)
    )


@app.command()
def merge_annotations(config_file: str):
    import spatialdata
    from spatialdata.models import TableModel
    import pandas

    config = tomllib.load(open(config_file, "rb"))

    sdata_path = config.get("spatialdata_path")
    annotation_path = config.get("annotation_df_path")
    save_path = config.get("save_path")
    assert save_path is not None, "Must provide save_path. Got None."

    sdata = spatialdata.read_zarr(sdata_path)
    annotation_df = pandas.read_csv(annotation_path, low_memory=False)

    adata = sdata.tables["table"].copy()

    adata = _merge_annotations(
        adata = adata,
        annotation_df = annotation_df,
        merge_cols = config.get("merge_cols"),
        on_col = config.get("on_col"),
        subset_filter = config.get("subset_filter"),
    )

    sdata.tables["table"] = TableModel.parse(adata)

    sdata.write(save_path, overwrite=True)

@app.command()
def combine_annotations(config_file: str):
    from spatialdata.models import ShapesModel
    config = tomllib.load(open(config_file, "rb"))

    sdata = spatialdata.read_zarr(conifg.get("sdata_path"))

    sdata = combine_sdata_annotations(
        sdata,
        incoming_shapes_key = config.get("incoming_shapes_key"),
        existing_shapes_key = config.get("existing_shapes_key"),
        overlap_drop_threshold = config.get("overlap_drop_threshold"),
        combined_shapes_key=config.get("combined_shapes_key"),
    )

    sdata.write(
        config.get("sdata_path"),
        overwrite=config.get("overwrite", False)
    )

@app.command()
def download_omero_roi(config_file: str):
    from histology_features.annotation import get_omero_roi

    config = tomllib.load(open(config_file, "rb"))

    get_omero_roi(
        image_id=config["image_id"],
        omero_id=config["omero_id"],
        omero_password=os.environ.get("OMERO_PASSWORD"),
        host=config["host"],
        save_dir=config["save_dir"], 
    )

@app.command()
def convert_to_zarr(config_file: str):
    """
    Read a Xenium Ranger output as a SpatialData
    object and save as zarr.

    Zarr will be saved inside the Xenium outs, like so:
    xenium_outs/
        ├── xenium_outs.zarr # SpatialData zarr created
        ├── experiment.xenium
        └── transcripts.zarr.zip
    """
    from spatialdata_io import xenium
    config = tomllib.load(open(config_file, "rb"))

    for xe_path in config.get("xenium_paths"):
        xe_path = pathlib.Path(xe_path)
        save_path = xe_path / f"{xe_path.stem}.zarr"
        print("Saving to:", save_path)

        sdata = xenium(xe_path)
        sdata.write(save_path, overwrite=config.get("overwrite", False))

@app.command()
def relate_omero_annotations(config_file: str):
    import spatialdata

    config = tomllib.load(open(config_file, "rb"))

    sdata = spatialdata.read_zarr(config.get("spatialdata_path"))

    sdata = relate_annotations(
        sdata,
        original_shapes_key = config.get("original_shapes_key"),
        new_shapes_key = config.get("new_shapes_key"),
        new_shapes_roi_name_col = config.get("new_shapes_roi_name_col"),
        table_key = config.get("table_key"),
        relation_method = config.get("relation_method"),
        multiple_new_shapes_method = config.get("multiple_new_shapes_method"),
        overlap_threshold=config.get("overlap_threshold", 0.8)
,    )

    if config.get("save_path").endswith(".h5ad"):
        sdata.tables[config.get("table_key")].write_h5ad(
            config.get("save_path")
        )
    else:
        sdata.write(
            config.get("save_path"),
            overwrite = config.get("overwrite", False)
        )

@app.command()
def batch_relate_omero_annotations(config_file: str):
    import spatialdata

    config = tomllib.load(open(config_file, "rb"))

    sdata = spatialdata.read_zarr(config.get("spatialdata_path"))

    table_key = config.get("table_key")

    batch_key = config.get("batch_key")

    batch_names = sdata[table_key].obs[batch_key].unique().tolist()

    related_keys = relate_keys(
        sdata.shapes.keys(),
        batch_names, 
    )

    assert len(related_keys) > 1, "No related keys found."

    print(f"Related shape keys: {related_keys}")

    # Get the indices for each batch
    batched_data = sdata[table_key].obs.groupby(batch_key).indices

    for batch_id, related_shape_names in related_keys.items():
        batch_idx = batched_data[batch_id]
        sdata = relate_annotations(
            sdata,
            original_shapes_key = related_shape_names[1],
            new_shapes_key = related_shape_names[0],
            new_shapes_roi_name_col = config.get("new_shapes_roi_name_col"),
            table_key = config.get("table_key"),
            relation_method = config.get("relation_method"),
            multiple_new_shapes_method = config.get("multiple_new_shapes_method"),
            batch_idx=batch_idx,
            overlap_threshold=config.get("overlap_threshold", 0.8)
        )

    if config.get("save_path").endswith(".h5ad"):
        sdata.tables[config.get("table_key")].write_h5ad(
            config.get("save_path")
        )
    else:
        sdata.write(
            config.get("save_path"),
            overwrite = config.get("overwrite", False)
        )
