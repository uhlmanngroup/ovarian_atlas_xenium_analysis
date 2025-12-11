from histology_features.polygons import sjoin_with_overlap_threshold

def relate_annotations(
    sdata,
    original_shapes_key: str,
    new_shapes_key: str,
    new_shapes_roi_name_col: list[str] | str,
    table_key: str = "table",
    relation_method: list["intersects", "overlaps", "both"] = "both",
    overlap_threshold: float = 0.8,
    multiple_new_shapes_method: list["keep", "first"] = "keep",
    batch_idx = slice(None),
):
    """Relate new annotations (for example, from Omero) 
    with existing Xenium segmentation shapes. Add the 
    new_shapes_key label ID in an obs column in the 
    AnnData object.

    Args:
        sdata (_type_): Input SpatialData object that contains all 
            annotations as shapes.
        original_shapes_key (str): Key for the original segmentation shapes
        new_shapes_key (str): Key for the incoming segmentations 
        new_shapes_roi_name_col (str): Name of the column in new_shapes GeoDataFrame
            to be added to adata.obs.
        table_key (str): Key of the table
    """

    # Convert single string to list for uniform processing
    if isinstance(new_shapes_roi_name_col, str):
        new_shapes_roi_name_col = [new_shapes_roi_name_col]

    if relation_method.casefold() == "both":
        relation_method = ["intersects", "overlaps"]
    else:
        relation_method = [relation_method]

    original_shapes = sdata.shapes[original_shapes_key]
    new_shapes = sdata.shapes[new_shapes_key]

    for rm in relation_method:

        if rm.casefold() == "overlaps":
            related_shapes = sjoin_with_overlap_threshold(
                original_shapes, 
                new_shapes, 
                overlap_threshold=overlap_threshold,
                overlap_method="left",
            )

        elif rm.casefold() == "intersects":
            related_shapes = original_shapes.sjoin(
                new_shapes,
                predicate=rm,
            )

        # Process each column separately
        for roi_col in new_shapes_roi_name_col:
            new_col_name = f"{roi_col}_{rm}"
            
            # Create a copy to work with for this specific column
            column_related_shapes = related_shapes.copy()

            if multiple_new_shapes_method.casefold() == "keep":
                # Create aggregation dictionary for this specific column
                agg_dict = {
                    "geometry": "first",
                    roi_col: lambda x: ", ".join([str(val) for val in set(x.dropna())]),  # remove duplicates and NaNs
                }
                column_related_shapes = column_related_shapes.groupby(column_related_shapes.index).agg(agg_dict)
            elif multiple_new_shapes_method.casefold() == "first":
                column_related_shapes = column_related_shapes.groupby(column_related_shapes.index).first()
            else:
                raise ValueError(f"multiple_new_shapes_method {multiple_new_shapes_method} not recognised.")

            column_related_shapes = column_related_shapes.rename(
                columns={
                    roi_col: new_col_name,
                    }
            )

            # Keep just the current ROI column
            column_related_shapes = column_related_shapes.loc[:, [new_col_name]]

            # Get the batch anndata.obs
            obs = sdata.tables[table_key].obs.iloc[batch_idx]

            # Since we subset the SpatialData column to the current batch,
            # after the first batch we have already added the column. So, 
            # drop it so we can add the column for this batch only.
            # Overall, I don't like this approach. This should be refactored
            # to create batch merges independently and then merge all at the same 
            # time (ie. merge once vs merge batch N times).
            if new_col_name in obs.columns:
                obs = obs.drop(new_col_name, axis=1)

            # Merge ROI names to the batch anndata subset
            obs_merged = obs.merge(column_related_shapes, left_on="cell_id", right_index=True, how="left")

            # Validate index matches
            assert obs.index.equals(obs_merged.index), "Index mismatch."

            sdata.tables[table_key].obs.loc[
                    sdata.tables[table_key].obs.index[batch_idx], new_col_name
                ] = obs_merged[new_col_name]

            # Encode as string for saving
            sdata.tables[table_key].obs[new_col_name] = sdata.tables[table_key].obs[new_col_name].fillna("NaN")
    
    return sdata
