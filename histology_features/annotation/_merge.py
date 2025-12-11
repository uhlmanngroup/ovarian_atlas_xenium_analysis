import anndata
import pandas
import logging

log = logging.getLogger(__name__)

def merge_annotations(
    adata: anndata.AnnData,
    annotation_df: pandas.DataFrame,
    merge_cols: str | list[str],
    on_col: str | list[str],
    subset_filter: dict[str, str] | list[dict[str, str]] | None = None,
    fill_nan_value: int | str = "nan",
) -> anndata.AnnData:
    """
    Merge annotations from a pandas DataFrame to an AnnData object.

    This function merges annotation data from `annotation_df` into `adata.obs` 
    based on a common column. Optionally, it can filter the annotation DataFrame 
    before merging.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object whose `.obs` DataFrame will be updated with annotations.
    annotation_df : pandas.DataFrame
        A DataFrame containing annotations to merge into `adata.obs`.
    merge_cols : str or list[str]
        Column(s) from `annotation_df` to merge into `adata.obs`.
    on_col : str
        The column name present in both `adata.obs` and `annotation_df` used for merging.
    subset_filter : dict[str, str] or list[dict[str, str]] or None, optional
        A dictionary or list of dictionaries specifying filter conditions for `annotation_df`
        before merging. Each dictionary should have `{column_name: value}` pairs. Values can 
        be a list.

    Returns
    -------
    anndata.AnnData
        The updated AnnData object with merged annotations in `adata.obs`.
    """

    if subset_filter is not None:
        annotation_df = subset_dataframe(annotation_df, subset_filter = subset_filter)

    merge_cols = merge_cols if isinstance(merge_cols, list) else [merge_cols]
    on_cols = on_col if isinstance(on_col, list) else [on_col]

    assert any(i in merge_cols for i in on_col), f"All elements of on_col ({on_col}) must be in merge_cols {merge_cols}"

    # Create temporary lowercase versions of the merge columns
    temp_cols = [col + '_lower' for col in on_cols]
    temp_mapping = dict(zip(temp_cols, on_cols))

    # Add lowercase versions to adata.obs
    for temp_col, orig_col in temp_mapping.items():
        adata.obs[temp_col] = adata.obs[orig_col].str.lower()

    # Create temporary annotation dataframe with lowercase merge columns
    annotation_df_temp = annotation_df.copy()
    for temp_col, orig_col in temp_mapping.items():
        annotation_df_temp[temp_col] = annotation_df_temp[orig_col].str.lower()

    cols_to_merge = [col for col in merge_cols if col not in adata.obs.columns or col in on_cols]
    mc = merge_cols + temp_cols

    cols_to_include = [col for col in merge_cols + temp_cols 
                   if col not in adata.obs.columns or col in temp_cols]

    # Perform the merge using the lowercase columns
    adata.obs = adata.obs.merge(
        annotation_df_temp[cols_to_include],
        on=temp_cols, 
        how="left"
    )

    # Clean up the temporary columns
    adata.obs = adata.obs.drop(columns=temp_cols)

    # Store for ensuring data isn't erroneously subset/duplicated
    initial_adata_len = len(adata)

    # Anndata merge with annotations can give rise to object columns 
    # with nan values. Convert these to str, which plays nice with 
    # SpatialData and zarr.
    for column in adata.obs.select_dtypes(include=[object]).columns:
        adata.obs[column] = adata.obs[column].astype(str)

    # Check adata has/hasn't duplicated/subset
    assert len(adata) == initial_adata_len, f"Length has changed from {initial_adata_len} to {len(adata)} after merge"

    return adata


def subset_dataframe(df: pandas.DataFrame, subset_filter: dict[str, str] | list[dict[str, str]]) -> pandas.DataFrame:
    """
    Subsets a DataFrame based on a dictionary of filtering conditions.
    
    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        subset_filter (dict): A dictionary where keys are column names (str),
                        and values are the values that should be in the respective columns.
    
    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    # Start with all True values (ie. all rows kept)
    if isinstance(subset_filter, dict):
        subset_filter = [subset_filter]  # Convert single dictionary to list
    
    combined_mask = pandas.Series(False, index=df.index)  # Start with all False values to apply OR condition
    
    for filter_dict in subset_filter:
        mask = pandas.Series(True, index=df.index)  # Start with all True values for AND condition
        
        for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, list):  # If value is a list, check if any match
                    value = [v.lower() if isinstance(v, str) else v for v in value]
                    col_values = df[key].astype(str).str.lower() if df[key].dtype == 'O' else df[key]
                    mask &= col_values.isin(value)
                else:  # Single value case
                    value = value.lower() if isinstance(value, str) else value
                    col_values = df[key].astype(str).str.lower() if df[key].dtype == 'O' else df[key]
                    mask &= col_values == value
        
        combined_mask |= mask  # Combine masks using OR condition
    
    return df[combined_mask]