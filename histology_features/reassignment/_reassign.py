from spatialdata import SpatialData
import geopandas
from spatialdata_io import xenium
import subprocess
from histology_features.spec import Constants, Xenium
from pathlib import Path
import os
import spatialdata_plot
import matplotlib.pyplot as plt
from spatialdata.models import ShapesModel

def transcript_reassignment(
    sdata: SpatialData,
    new_shapes_key: str,
    xenium_bundle_path: str,
    xenium_ranger_path: str,
    sdata_save_path: str,
    plot: bool = False,
):
    # We can't choose the path Xenium Ranger saves to, so change
    # the current working directory to where we wish to save 
    # xenium ranger output
    os.chdir(xenium_bundle_path)

    new_segmentations = sdata.shapes[new_shapes_key]
    new_segmentations = new_segmentations[new_segmentations["roi_name"] != "oocyte"]

    # Save segmentations to disk in cache
    segmentation_path = Path(Constants.histology_features_cache, "new_segmentations.geojson")
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)

    new_segmentations.to_file(
        segmentation_path,
        driver="GeoJSON",
    )

    cmd = [
        xenium_ranger_path, "import-segmentation",
        f"--id={Path(xenium_bundle_path).parent.name}",
        f"--xenium-bundle={xenium_bundle_path}",
        f"--cells={segmentation_path}",
        f"--units=pixels",
        f"--localcores=32",
        f"--localmem=128"
    ]

    # Execute the command
    # subprocess.run(cmd, check=True)

    recomputed_sdata = xenium(
        Path(xenium_bundle_path, Path(xenium_bundle_path).parent.name, "outs"),
        cells_boundaries = True,
        cells_table = True,
        nucleus_boundaries = False,
        cells_as_circles = False,
        cells_labels = False,
        nucleus_labels = False,
        transcripts = False,
        morphology_mip = True,
        morphology_focus = True,
        aligned_images = False,
    )

    recomputed_cell_boundaries = recomputed_sdata.shapes["cell_boundaries"]

    # The new_segmentations are in the pixel coordinate system.
    # However, Xenium Ranger saves polygons in the micron range. 
    # So, convert our 
    new_segmentations["geometry"] = new_segmentations.scale(
        Xenium.pixel_size.value, 
        Xenium.pixel_size.value, 
        origin=(0, 0)
    )

    recomputed_cell_boundaries = transfer_names_by_overlap(
        new_segmentations,
        recomputed_cell_boundaries,
        name_column="roi_name",
        threshold=0.95,
    )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        new_segmentations.plot(ax=ax, color=None, edgecolor="black", alpha=0.5, aspect=1)
        recomputed_cell_boundaries.plot(ax=ax, color="blue", edgecolor=None, alpha=0.5, aspect=1)
        fig.savefig("geo_plot.png")

    # Update the cell boundaries returned by Xenium Ranger with the ROI names for the 
    # omero annotation
    recomputed_sdata.shapes["cell_boundaries"] = ShapesModel.parse(recomputed_cell_boundaries)

    recomputed_sdata.write(sdata_save_path, overwrite=True)

def transfer_names_by_overlap(source_gdf, target_gdf, name_column, threshold):
    """
    Transfer the name_column column from source GeoDataFrame to target GeoDataFrame based on polygon overlap.
    
    Parameters:
    -----------
    source_gdf : GeoDataFrame
        Source GeoDataFrame containing the name_column column to be transferred.
    target_gdf : GeoDataFrame
        Target GeoDataFrame that will receive the name values.
    name_column: str
        Column containing a string to be transferred.
    threshold : float, default 0.5
        Minimum overlap ratio required to transfer the name.
        A value of 0.5 means that at least 50% of the target polygon
        must be covered by the source polygon.
    
    Returns:
    --------
    GeoDataFrame
        A copy of target_gdf with an additional 'transferred_name' column.
    """
    # Verify that both inputs are GeoDataFrames
    if not isinstance(source_gdf, geopandas.GeoDataFrame) or not isinstance(target_gdf, geopandas.GeoDataFrame):
        raise TypeError("Both source_gdf and target_gdf must be GeoDataFrames")
    
    # Verify that source_gdf has a name_column column
    if name_column not in source_gdf.columns:
        raise ValueError("Source GeoDataFrame must have a name_column column")

    # Create a copy of the target GeoDataFrame to not modify the original
    result_gdf = target_gdf.copy()
    
    # Create a new column for transferred names
    result_gdf['transferred_name'] = None
    
    # For each polygon in the target GeoDataFrame
    for idx, target_row in result_gdf.iterrows():
        target_geom = target_row.geometry
        
        if target_geom is None or target_geom.is_empty:
            continue
            
        target_area = target_geom.area
        if target_area <= 0:
            continue
            
        # Find all source polygons that intersect with this target polygon
        potential_matches = source_gdf[source_gdf.intersects(target_geom)]
        
        if len(potential_matches) == 0:
            continue
            
        # Calculate overlap area for each potential match
        overlaps = []
        for _, source_row in potential_matches.iterrows():
            source_geom = source_row.geometry
            
            if source_geom is None or source_geom.is_empty:
                continue
                
            # Calculate the intersection
            intersection = target_geom.intersection(source_geom)
            
            # Calculate the overlap ratio (intersection area / target area)
            overlap_ratio = intersection.area / target_area
            
            overlaps.append({
                name_column: source_row[name_column],
                'overlap_ratio': overlap_ratio
            })
        
        if not overlaps:
            continue
            
        # Find the source polygon with maximum overlap
        max_overlap = max(overlaps, key=lambda x: x['overlap_ratio'])
        
        # Only transfer the name if the overlap exceeds the threshold
        if max_overlap['overlap_ratio'] >= threshold:
            result_gdf.at[idx, 'transferred_name'] = max_overlap[name_column]
    
    return result_gdf