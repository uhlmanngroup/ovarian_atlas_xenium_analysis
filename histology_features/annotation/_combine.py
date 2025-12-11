from spatialdata import SpatialData
from spatialdata.models import (
    ShapesModel,
)
import typing
import geopandas
import shapely
import pandas
import numpy

def combine_sdata_annotations(
    sdata: typing.Union[SpatialData, str],
    incoming_shapes_key: str,
    existing_shapes_key: str,
    overlap_drop_threshold: float,
    combined_shapes_key: str,
):
    """
    Combine shapes in a SpatialData object.

    incoming_shapes_key represents annotations that 
    are derived from a non-xenium source, such as Omero
    annotations.

    existing_shapes_key are the cell_boundary segmentations
    defined by Xenium Ranger.

    Set overlap_drop_threshold > 1 to ensure all incoming_shapes_key
    are preserved. Any overlapping polygons will be dropped.
    """
    if isinstance(sdata, str):
        sdata = SpatialData.read_zarr(sdata)

    incoming_shapes = sdata.shapes[incoming_shapes_key]
    existing_shapes = sdata.shapes[existing_shapes_key]

    new_shapes = combine_new_polygons(
        incoming_shapes, 
        existing_shapes, 
        overlap_drop_threshold=overlap_drop_threshold,
    )

    sdata.shapes[combined_shapes_key] = ShapesModel.parse(new_shapes)

    return sdata
    
def combine_new_polygons(
    new_polygons, existing_polygons, overlap_drop_threshold: float, return_dropped: bool = False,
):
    """
    Combine a new set of polygons with an existing set. This is intended to allow merging
    of a corrected subset of polygons (eg. expert annotation) with an existing set
    (eg. automatic segmentation).

    This function will drop existing polygons that are overlapped by the new polygons
    above the overlap_drop_threshold.

    For overlapping polygons that are not dropped, they will have their geometires changed
    to exclude the new polygon region. That is, the new polygon geometry is subtracted from
    the existing polygon geometry.
    """

    if isinstance(new_polygons, list):
        new_polygons = geopandas.DataFrame({"geometry": new_polygons})

    # Find which existing polygons overlap with new polygons.
    # The returned GDF will have a geometry column for existing
    # polygons and column index_right for the index to the overlapping
    # polygons found in new_polygons
    overlapping = geopandas.sjoin(
        existing_polygons, new_polygons, how="inner", predicate="intersects"
    )

    if not overlapping.empty:
        overlapping["overlap"] = overlapping.apply(
            lambda x: measure_polygon_overlap(
                polygon_b=x["geometry"],
                polygon_a=new_polygons.loc[x["index_right"], "geometry"],
            ),
            axis=1,
        )

        # If the overlap is above a certain threshold, drop the polygon
        drop_idx = overlapping[overlapping["overlap"] >= overlap_drop_threshold].index
        print(f"Dropping {len(drop_idx)} polyons.")
        # Drop overlapping from both dataframes. We will modify the permitted
        # polygons in overlapping and transfer these to existing polygons
        overlapping = overlapping.drop(drop_idx)
        existing_polygons = existing_polygons.drop(drop_idx)

        if return_dropped: 
            inverse_idx = numpy.setdiff1d(overlapping.index, drop_idx)
            dropped_polygons = overlapping.drop(inverse_idx)

        # All overlapping polygons have been removed, so return 
        # the concatenation of non-dropped existing polygons concatenated
        # with the new polygons
        if len(overlapping) == len(drop_idx):
            out_polygons = pandas.concat([existing_polygons, new_polygons], axis=0)
            if return_dropped:
                return out_polygons, overlapping
            else:
                return out_polygons



        # For polygons that did not meet the threshold (ie. not dropped), find the
        # difference with their overlapping polygon. The goal here is to remove
        # the region of the polygon in existing_polygon, which will be replaced
        # in space by the new_polygon
        overlapping["geometry"] = overlapping.apply(
            lambda x: x["geometry"].difference(
                new_polygons.loc[x["index_right"], "geometry"]
            ),
            axis=1,
        )

        overlapping["index"] = overlapping.index

        overlapping = shape_intersection_aggregate(
            gdf=overlapping,
            group_column="index",
        )

        # Now update existing_polygons with the updated overlapped polygons
        existing_polygons.loc[overlapping.index, "geometry"] = overlapping["geometry"]
    else:
        dropped_polygons = None

    out_polygons = pandas.concat([existing_polygons, new_polygons], axis=0)

    if return_dropped:
        return out_polygons, dropped_polygons
    else:
        return out_polygons


def shape_intersection_aggregate(
    gdf: geopandas.GeoDataFrame,
    group_column: str = None,
    geometry_column: str = "geometry",
):
    if group_column is None:
        overlapping = gdf["geometry"].intersection_all()
        overlapping = geopandas.GeoDataFrame({"geometry": [overlapping]})
    else:
        # Multiple new_polygons may touch the existing polygons, leading
        # to a single existing_polygon needing to be reshaped multiple times.
        # Use the intersection to determine what the existing polygon should become.
        # Dissolve achieves this.

        # Creates a GeoSeries
        overlapping = gdf.groupby(group_column)["geometry"].aggregate(
            shapely.intersection_all
        )
        overlapping = geopandas.GeoDataFrame(overlapping)

    # Explode the multipolygons to single polygons
    overlapping = overlapping.explode(column="geometry")
    # Delete LineStrings
    overlapping = overlapping[overlapping["geometry"].geom_type != "LineString"]

    return overlapping

def measure_polygon_overlap(
    polygon_a: shapely.Polygon, polygon_b: shapely.Polygon
) -> float:
    """Measure the percentage that polygon_a overlaps the area of polygon_b.
    This is measured as a percentage of **polygon_b**'s area.

    Args:
        polygon_a (shapely.Polygon): Overlapping polygon
        polygon_b (shapely.Polygon): Polygon being overlapped

    Returns:
        float: Percentage of polygon b that is overlapped by a (0, 1).
    """

    overlap_area = polygon_a.intersection(polygon_b).area

    polygon_b_overlap_perc = overlap_area / polygon_b.area

    return polygon_b_overlap_perc
