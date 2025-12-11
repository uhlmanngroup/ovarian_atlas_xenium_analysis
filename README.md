# histology_features

Tools for extracting features from histology images and working with spatial transcriptomics data.

## Installation

```bash
pip install -e .
```

Or, use the Docker container. *Recommended*.

## CLI Usage

After installation, the CLI is available as `histology_features`. All commands accept a TOML configuration file:

```bash
histology_features <command> /path/to/config.toml
```

### Commands

#### `convert_to_zarr`

Convert Xenium Ranger output to SpatialData zarr format.

```toml
xenium_paths = ["/path/to/xenium_outs"]
overwrite = false
```

The zarr is saved inside the Xenium output directory as `xenium_outs.zarr`.

---

#### `concatenate`

Concatenate multiple SpatialData objects into one.

```toml
[concatenate]
file_paths = [
    "/path/to/sample1.zarr",
    "/path/to/sample2.zarr"
]
save_path = "/path/to/output.zarr"
table_only = false        # Optional: only concatenate tables
drop_cols = ["col1"]      # Optional: columns to drop
output_obs_csv = false    # Optional: Also export AnnData obs as CSV
```

---

#### `submit_registration`

Register and align multiple histology images using VALIS (other registration methods can be easily implemented, see `histology_features.regisration._registration`).

Alignment cache and low resolution alignments for visual inspection while determining optimal `masking_kwargs` and `alignment_kwargs` are saved in `save_path/.histology_features_cache/`. Check the `overlaps` directory to see how well rigid/non-rigid registration has performed.

```toml
# Define image specific arguments within [[image]] tags
[[image]]
path = "/path/to/image1.tiff"
target_image = true  # Mark one image as the target
mask_ids = ["S10", "S13"] # Identify all ROIs in the histology image
mask_to_align = ["S10", "S13"] # Then identify which ROIs to actually align
source_image_mask_order_method = "left_right" # How the sections are ordered when mask_ids were defined
source_image_mask_downsample_scale = [1, 32, 32] # C Y X. Scaled down images are faster to process masks for
masking_kwargs = {manual_threshold = 20}

[[image]]
path = "/path/to/image2.tiff"

# These are the overarching alignment args for all images.
[alignment] 
save_path = "/path/to/output.zarr"
transformation = "both" # Options: "rigid", "non-rigid", "both"
save_full_slide = false # When determining optimal registration params, set to false for speed. There is no need to write full image until a decent registration has been found)
overwrite = false
dimensions = {"z" = 0, "y" = 1, "x" = 2, "c" = 3} # Define dimension order if not CYX

# max_processed_image_dim_px is useful for aligning consecutive slides. A lower value (eg. 1000) 
# can allow better initial matching of similar (ie. not same) sections.
# If aligning matching sections (ie. DAPI + H&E of the same cells), a higher max_processed_image_dim_px
# can be better (eg. 2_000)
alignment_kwargs = {max_non_rigid_registration_dim_px=1_000, max_processed_image_dim_px=1_000}
```

---

#### `add_batch_column`

Add a batch identifier column to a SpatialData table. This is used by the NextFlow pipeline to add a batch column. The z is defined in the NextFlow samplesheet.

```toml
spatialdata_path = "/path/to/input.zarr"
batch_id = "${sample_id}"
save_path = "/path/to/output.zarr"
overwrite = true
```

---

#### `merge_annotations`

Merge annotation data from a CSV into a SpatialData table. This is used to merge annotations from the batch_corrected and annotated 

```toml
spatialdata_path = "/path/to/input.zarr"
annotation_df_path = "/path/to/annotations.csv" # This is an annotated anndata.obs
save_path = "/path/to/output.zarr"
merge_cols = ["annotation_col1", "annotation_col2"] # eg. broad_annotation and fine_annotation
on_col = "cell_id"

# A dictionary or list of dictionaries specifying filter conditions for `annotation_df`
# before merging. Each dictionary should have `{column_name: value}`. Optional.
subset_filter = {} 
```

---

#### `download_omero_roi`

Download ROI annotations from an OMERO server. **Requires your OMERO password is available as environmental variable `$OMERO_PASSWORD`**

```toml
image_id = 12345
omero_id = "username"
host = "omero.server.com"
save_dir = "/path/to/save"
```

---

#### `relate_omero_annotations`

Relate OMERO annotations (shapes) to cells in a SpatialData object.

```toml
spatialdata_path = "/path/to/input.zarr"
original_shapes_key = "cell_circles"
new_shapes_key = "omero_rois"
new_shapes_roi_name_col = "roi_name"
table_key = "table"
relation_method = "contains" 
multiple_new_shapes_method = "first" 
save_path = "/path/to/output.zarr"  # or .h5ad
overwrite = false
```

---

#### `relate_omero_annotations`

Relate OMERO annotations to cells across multiple batches.

```toml
spatialdata_path = "/path/to/input.zarr"
table_key = "table"
batch_key = "batch" # Column used to identify batches 
new_shapes_roi_name_col = "roi_name"
relation_method = "contains" # "intersects", "overlaps", or "both". Intersects keeps all annotations that intersect. Overlaps only keeps annotations that are above overlap_threshold.
multiple_new_shapes_method = "first" # "keep" or "first". Keep will keep all overlapping/intersecting annotations, whereas first will only keep the first.
overlap_threshold = 0.8 # Only used if `relation_method=='overlaps'`. 
save_path = "/path/to/output.zarr"
overwrite = true
```