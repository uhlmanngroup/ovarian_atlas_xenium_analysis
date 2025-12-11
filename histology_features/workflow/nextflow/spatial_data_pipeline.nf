process add_batch_column {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_batch_column.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_batch_column.toml << EOF
spatialdata_path = "${params_map.path}"
batch_id = "${sample_id}"
save_path = "${sample_id}_batch_column.zarr"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/histology_features add-batch-column ${sample_id}_batch_column.toml
    """
}

process merge_annotations_scvi {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(batch_id_file)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_merge_annotations_scvi.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_merge_annotations_scvi_config.toml << EOF
spatialdata_path = "${batch_id_file}"
annotation_df_path = "${params_map.merge_annotations_scvi_annotation_df_path}"
merge_cols = ${params_map.merge_annotations_scvi_merge_cols}
on_col = ${params_map.merge_annotations_scvi_on_col}
save_path = "${sample_id}_merge_annotations_scvi.zarr"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/histology_features merge-annotations ${sample_id}_merge_annotations_scvi_config.toml
    """
}

process merge_annotations_dot {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(annotated_scvi)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_merge_annotations_dot.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_merge_annotations_dot_config.toml << EOF
spatialdata_path = "${annotated_scvi}"
annotation_df_path = "${params_map.merge_annotations_DOT_annotation_df_path}"
merge_cols = ${params_map.merge_annotations_DOT_merge_cols}
on_col = ${params_map.merge_annotations_DOT_on_col}
save_path = "${sample_id}_merge_annotations_dot.zarr"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/histology_features merge-annotations ${sample_id}_merge_annotations_dot_config.toml
    """
}

process relate_omero_annotations {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(annotated_dot)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_omero_annotations.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_relate_omero.toml << EOF
spatialdata_path = "${annotated_dot}"
save_path = "${sample_id}_omero_annotations.zarr"

original_shapes_key = "${params_map.original_shapes_key}" # This is xenium seg
new_shapes_key = "${params_map.new_shapes_key}" # This is omero annotations
new_shapes_roi_name_col = ["roi_name", "roi_id"]
table_key = "table"
relation_method = "both"
multiple_new_shapes_method = "keep"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/histology_features relate-omero-annotations ${sample_id}_relate_omero.toml
    """
}

process spatial_axis_stromal_single {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(related_annotations)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_stromal_single.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_stromal_single_config.toml << EOF
data_path = "${related_annotations}"

annotation_column = "${params_map.spatial_axis_stromal_single_annotation_column}"
k_neighbours = ${params_map.spatial_axis_stromal_single_k_neighbours}
annotation_order = ${params_map.spatial_axis_stromal_single_annotation_order}
save_column = "${params_map.spatial_axis_stromal_single_save_column}"
save_path = "${sample_id}_spatial_axis_stromal_single.zarr"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_stromal_single_config.toml
    """
}

process spatial_axis_stromal_multiple {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(stromal_single)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_stromal_multiple.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_stromal_multiple_config.toml << EOF
data_path = "${stromal_single}"

annotation_column = "${params_map.spatial_axis_stromal_multiple_annotation_column}"
k_neighbours = ${params_map.spatial_axis_stromal_multiple_k_neighbours}
annotation_order = ${params_map.spatial_axis_stromal_multiple_annotation_order}
save_column = "${params_map.spatial_axis_stromal_multiple_save_column}"
save_path = "${sample_id}_spatial_axis_stromal_multiple.zarr"
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_stromal_multiple_config.toml
    """
}

process spatial_axis_growing {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(stromal_multiple)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_growing.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_growing_config.toml << EOF
data_path = "${stromal_multiple}"

annotation_column = "${params_map.spatial_axis_growing_annotation_column}"
k_neighbours = ${params_map.spatial_axis_growing_k_neighbours}
annotation_order = ${params_map.spatial_axis_growing_annotation_order}
save_column = "${params_map.spatial_axis_growing_save_column}"
save_path = "${sample_id}_spatial_axis_growing.zarr"
distance_threshold = ${params_map.spatial_axis_growing_distance_threshold}
distance_k_neighbors = ${params_map.spatial_axis_growing_distance_k_neighbors}
scaling_factor = ${params_map.spatial_axis_growing_scaling_factor}
normalise = ${params_map.spatial_axis_growing_normalise}
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_growing_config.toml
    """
}


process spatial_axis_atretic {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(growing_spatial_axis)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_atretic.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_atretic_config.toml << EOF
data_path = "${growing_spatial_axis}"

annotation_column = "${params_map.spatial_axis_atretic_annotation_column}"
k_neighbours = ${params_map.spatial_axis_atretic_k_neighbours}
annotation_order = ${params_map.spatial_axis_atretic_annotation_order}
save_column = "${params_map.spatial_axis_atretic_save_column}"
save_path = "${sample_id}_spatial_axis_atretic.zarr"
distance_threshold = ${params_map.spatial_axis_atretic_distance_threshold}
distance_k_neighbors = ${params_map.spatial_axis_atretic_distance_k_neighbors}
scaling_factor = ${params_map.spatial_axis_atretic_scaling_factor}
normalise = ${params_map.spatial_axis_atretic_normalise}
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_atretic_config.toml
    """
}

process spatial_axis_growing_knn1 {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(atretic_spatial_axis)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_growing_knn1.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_growing_knn1_config.toml << EOF
data_path = "${atretic_spatial_axis}"

annotation_column = "${params_map.spatial_axis_growing_annotation_column}"
k_neighbours = ${params_map.spatial_axis_growing_k_neighbours_knn1}
annotation_order = ${params_map.spatial_axis_growing_annotation_order}
save_column = "${params_map.spatial_axis_growing_knn1_save_column}"
save_path = "${sample_id}_spatial_axis_growing_knn1.zarr"
distance_threshold = ${params_map.spatial_axis_growing_knn1_distance_threshold}
distance_k_neighbors = ${params_map.spatial_axis_growing_knn1_distance_k_neighbors}
scaling_factor = ${params_map.spatial_axis_growing_knn1_scaling_factor}
normalise = ${params_map.spatial_axis_growing_knn1_normalise}
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_growing_knn1_config.toml
    """
}

process spatial_axis_atretic_knn1 {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(growing_knn1_spatial_axis)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_spatial_axis_atretic_knn1.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_spatial_axis_atretic_knn1_config.toml << EOF
data_path = "${growing_knn1_spatial_axis}"

annotation_column = "${params_map.spatial_axis_atretic_annotation_column}"
k_neighbours = ${params_map.spatial_axis_growing_k_neighbours_knn1}
annotation_order = ${params_map.spatial_axis_atretic_annotation_order}
save_column = "${params_map.spatial_axis_atretic_knn1_save_column}"
save_path = "${sample_id}_spatial_axis_atretic_knn1.zarr"
distance_threshold = ${params_map.spatial_axis_atretic_knn1_distance_threshold}
distance_k_neighbors = ${params_map.spatial_axis_atretic_knn1_distance_k_neighbors}
scaling_factor = ${params_map.spatial_axis_atretic_knn1_scaling_factor}
normalise = ${params_map.spatial_axis_atretic_knn1_normalise}
EOF

    # Run your CLI with the generated TOML
    ~/.local/bin/spatial_axis ${sample_id}_spatial_axis_atretic_knn1_config.toml
    """
}

process add_embedding {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(atretic_knn1_spatial_axis)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_with_embeddings.zarr")
    
    script:
    """
    # Generate TOML config
    cat > ${sample_id}_add_embedding.toml << EOF
spatialdata_path = "${atretic_knn1_spatial_axis}"
embedding_path = "${params_map.embedding_path}"
save_path = "${sample_id}_with_embeddings.zarr"
EOF

    ~/.local/bin/histology_features add-embedding ${sample_id}_add_embedding.toml
    """
}

process rasterize_shapes {
    tag "$sample_id"
    
    input:
    tuple val(sample_id), val(params_map), path(input_zarr)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_rasterized.zarr")
    
    script:
    """
    # Create Python script
    cat > rasterize_script.py << 'EOF'
import spatialdata
from spatialdata import rasterize
from spatialdata.models import Labels2DModel

# Read the input zarr
sdata = spatialdata.read_zarr("${input_zarr}")

# Find the key containing "target_image_shapes"
rasterize_shapes_key = None
target_image_key = None
for key in sdata._shared_keys:
    if "target_image_shapes" in key:
        rasterize_shapes_key = key
    if "target_image" in key and "shapes" not in key:
        target_image_key = key

if rasterize_shapes_key is None:
    raise ValueError("No key containing 'target_image_shapes' found in sdata")

if target_image_key is None:
    raise ValueError("No key containing 'target_image' found in sdata")

# Rasterize the shapes
# This rasterises them as an image first since rasterizing them 
# directly to labels leads to a uint16 limit (~65k objects)
sdata["xenium_segmentations_image"] = rasterize(
    sdata[rasterize_shapes_key],
    ["x", "y"],
    min_coordinate=[0, 0],
    max_coordinate=[
        sdata[target_image_key].scale0.dims["x"], 
        sdata[target_image_key].scale0.dims["y"]
    ],
    target_coordinate_system="global",
    target_unit_to_pixels=1,
).astype(int)

# Set the attrs to hold the label index dictionary
sdata.attrs["label_index_to_category"] = sdata["xenium_segmentations_image"].label_index_to_category

# Now convert the labels image to a Labels object (this also avoids the uint16 limit).
# Unsure why it's even a thing.
sdata["xenium_segmentations"] = Labels2DModel.parse(
        sdata["xenium_segmentations_image"][0, ...]
    )

# Delete the label image
del sdata["xenium_segmentations_image"]

sdata["table"].obs["region"] = "xenium_segmentations"

sdata.set_table_annotates_spatialelement(
    table_name="table", 
    region="xenium_segmentations", 
    region_key="region", 
    instance_key="cell_id"
)

# Write the output
sdata.write("${sample_id}_rasterized.zarr")
EOF

    # Run the Python script
    python rasterize_script.py
    """
}

process rename_data {

    input:
    tuple val(sample_id), val(params_map), path(input_zarr)
    
    output:
    tuple val(sample_id), val(params_map), path("${sample_id}_renamed.zarr")

    script:
    """
    # Create Python script
    cat > rename_script.py << 'EOF'
import spatialdata

# Read the input zarr
sdata = spatialdata.read_zarr("${input_zarr}")

# Rename histology annotation column
COL_NAME = "roi_name_intersects"

if COL_NAME in sdata["table"].obs.columns:
    sdata["table"].obs["histology_annotation"] = sdata["table"].obs[COL_NAME]

sdata["post_xenium_histology"] = sdata["non_rigid_image_${sample_id}_histology"]

# Write the output
sdata.write("${sample_id}_renamed.zarr")
EOF

    # Run the Python script
    python rename_script.py
    """
}

process concatenate_zarr {
    publishDir "${params.outdir}", mode: 'copy'
    
    input:
    path zarr_files  // Will receive all .zarr files as a list
    
    output:
    path "all_donors_table_only.zarr"
    
    script:
    def zarr_list = zarr_files.collect { "  \"${it}\"" }.join(",\n")
    """
    # Generate TOML config
    cat > concatenate.toml << EOF
[concatenate]
file_paths = [${zarr_list}]
save_path = "all_donors_table_only.zarr"
table_only = true
output_obs_csv = true
EOF

    ~/.local/bin/histology_features concatenate concatenate.toml
    """
}


process publish_outputs {
    publishDir "${params.outdir}", mode: 'copy'
    
    input:
    tuple val(sample_id), val(params_map), path(files)
    
    output:
    path "*.zarr"
    
    script:
    // Use output_name from sample sheet if provided, otherwise fall back to sample_id
    output_name = params_map.output_name ?: "${sample_id}_final.zarr"
    """
    cp -r ${files} ${output_name}
    """
}

workflow {
    samples_ch = Channel
        .fromPath(params.samplesheet)
        .splitCsv(header: true, sep: '\t')
        .map { row -> 
            return [
                row.sample_id, 
                row,
            ]
        }

    // Regular processing pipeline (all samples)
    add_batch_column(samples_ch)
    rasterize_shapes(add_batch_column.out)
    merge_annotations_scvi(rasterize_shapes.out)
    merge_annotations_dot(merge_annotations_scvi.out)
    
    merge_annotations_dot.out.branch { sample_id, params_map, input_data ->
        to_relate: params_map.original_shapes_key != "False"
        pass_through: params_map.original_shapes_key == "False"
    }.set { relate_branch }

    related_annotations = relate_omero_annotations(relate_branch.to_relate)
    after_relation = related_annotations.mix(relate_branch.pass_through)

    // Sequential processing - each step builds on the previous
    
    // Step 1: Process stromal samples, pass through others unchanged
    after_relation.branch { sample_id, params_map, input_data ->
        stromal: params_map.run_stromal == "True"
        pass_through: params_map.run_stromal != "True"
    }.set { stromal_branch }

    stromal_single = spatial_axis_stromal_single(stromal_branch.stromal)
    stromal_multiple = spatial_axis_stromal_multiple(stromal_single)
    after_stromal = stromal_multiple.mix(stromal_branch.pass_through)

    // Step 2: Process growing samples
    after_stromal.branch { sample_id, params_map, input_data ->
        growing: params_map.run_growing == "True"
        pass_through: params_map.run_growing != "True"
    }.set { growing_branch }
    
    growing_processed = spatial_axis_growing(growing_branch.growing)
    after_growing = growing_processed.mix(growing_branch.pass_through)

    // Step 3: Process growing_knn1 samples (same condition as growing)
    after_growing.branch { sample_id, params_map, input_data ->
        growing_knn1: params_map.run_growing == "True"
        pass_through: params_map.run_growing != "True"
    }.set { growing_knn1_branch }
    
    growing_knn1_processed = spatial_axis_growing_knn1(growing_knn1_branch.growing_knn1)
    after_growing_knn1 = growing_knn1_processed.mix(growing_knn1_branch.pass_through)

    // Step 4: Process atretic samples
    after_growing_knn1.branch { sample_id, params_map, input_data ->
        atretic: params_map.run_atretic == "True"
        pass_through: params_map.run_atretic != "True"
    }.set { atretic_branch }
    
    atretic_processed = spatial_axis_atretic(atretic_branch.atretic)
    after_atretic = atretic_processed.mix(atretic_branch.pass_through)

    // Step 5: Process atretic_knn1 samples (same condition as atretic)
    after_atretic.branch { sample_id, params_map, input_data ->
        atretic_knn1: params_map.run_atretic == "True"
        pass_through: params_map.run_atretic != "True"
    }.set { atretic_knn1_branch }
    
    atretic_knn1_processed = spatial_axis_atretic_knn1(atretic_knn1_branch.atretic_knn1)
    after_atretic_knn1 = atretic_knn1_processed.mix(atretic_knn1_branch.pass_through)

    // Step 6: Add embedding if embedding_path is provided
    after_atretic_knn1.branch { sample_id, params_map, input_data ->
        has_embedding: params_map.embedding_path != null && params_map.embedding_path != "" && params_map.embedding_path != "null"
        pass_through: params_map.embedding_path == null || params_map.embedding_path == "" || params_map.embedding_path == "null"
    }.set { embedding_branch }
    
    embedding_processed = add_embedding(embedding_branch.has_embedding)
    all_results = embedding_processed.mix(embedding_branch.pass_through)

    // Rename parts of the data
    rename_data(all_results)

    final_results = publish_outputs(rename_data.out)

    final_results = final_results.collect()

    concatenate_zarr(final_results)
    
}