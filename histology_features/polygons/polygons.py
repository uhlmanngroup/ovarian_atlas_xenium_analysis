import numpy
import skimage
import shapely
import geopandas
import typing
import spatialdata_io
import pandas
import re

from histology_features.utility.utils import combine_dicts

def extract_xy(filename):
    """Sort based on XY positions"""
    pattern = r'_\d+_\d+'
    match = re.findall(pattern, filename)[0]
    return int(match.split("_")[1]), int(match.split("_")[2])

def translate_polygons_to_crop(
        polygons,
        x_min,
        y_min,
        pixel_size: int = 0.2125,
        geometry_col: str = "geometry"
):
    """Translate polygons to a new origin based upon the crop
    x_min and y_min."""
    # Work with a copy of the dataset to avoid set warnings
    polygons = polygons.copy()
    polygons[geometry_col] = polygons[geometry_col].apply(
            lambda x: shapely.affinity.translate(
                x, -x_min * pixel_size, -y_min * pixel_size
            )
        )
    
    return polygons

def get_labels_from_polygons(
    polygons,
    crop_window_size,
    pixel_size: float = 0.2125,
) -> numpy.array:
    """
    Rasterize labels from a GeoPandas DataFrame.

    If rasterising a cropped region, make sure that the coordinates
    have been translated to the new crop origin with translate_polygons_to_crop
    """
    output_array = numpy.zeros((crop_window_size, crop_window_size), dtype=numpy.int32)

    label_ids = numpy.arange(polygons.shape[0]) + 1

    # def vectorized_rasterize(polygon, label_id):
    #     # Color only the outlines
    #     exterior_coords = numpy.array(polygon.exterior.coords)
        
    #     # Rescale coords with the inverse of the pixel size (ie. convert from microns to pixels)
    #     exterior_coords = numpy.multiply(exterior_coords, pixel_size**-1)
    #     rr, cc = skimage.draw.polygon(
    #         exterior_coords[:, 1], exterior_coords[:, 0], shape=output_array.shape
    #     )
    #     output_array[rr, cc] = label_id

    # numpy.vectorize(vectorized_rasterize)(polygons, label_ids)

    for label_id, polygon in zip(label_ids, polygons):
        exterior_coords = numpy.array(polygon.exterior.coords)
        exterior_coords = numpy.multiply(exterior_coords, pixel_size**-1)
        rr, cc = skimage.draw.polygon(
            exterior_coords[:, 1], exterior_coords[:, 0], shape=output_array.shape
        )
        output_array[rr, cc] = label_id

    return output_array

def random_shapely_circles(
    image_shape, num_circles, min_radius=15, max_radius=35, seed=None, pixel_size=1
):
    """Generate a list of randomly sized shapely Polygons.

    pixel_size: This is used for testing as Xenium spatial transcriptomic
    datasets scale their segmentation polygons to the micron coordindate system.
    For Xenium, there are 0.2125 pixels per micron, so we scale to this
    """
    circles = []

    numpy.random.seed(seed=seed)

    for _ in range(num_circles):
        x_center = numpy.random.uniform(0, image_shape[1]) * pixel_size
        y_center = numpy.random.uniform(0, image_shape[0]) * pixel_size
        radius = numpy.random.uniform(min_radius, max_radius) * pixel_size

        circle = shapely.geometry.Point(x_center, y_center).buffer(radius)

        circles.append(circle)

    return circles

def random_shapely_squares(
    image_shape, num_squares, min_size=15, max_size=35, seed=None, pixel_size=1
):
    """Generate a list of randomly sized shapely Polygon squares.

    pixel_size: This is used for testing as Xenium spatial transcriptomic
    datasets scale their segmentation polygons to the micron coordindate system.
    For Xenium, there are 0.2125 pixels per micron, so we can scale to this
    """
    squares = []

    numpy.random.seed(seed=seed)

    for _ in range(num_squares):
        x_center = numpy.random.randint(0, image_shape[1]) * pixel_size
        y_center = numpy.random.randint(0, image_shape[0]) * pixel_size

        print(x_center, y_center)

        if min_size != max_size:
            side_length = numpy.random.randint(min_size, max_size)
        else: 
            side_length = min_size

        half_side = (side_length * pixel_size) / 2

        square = shapely.geometry.Polygon([
                (x_center - half_side, y_center - half_side),
                (x_center + half_side, y_center - half_side),
                (x_center + half_side, y_center + half_side),
                (x_center - half_side, y_center + half_side)
            ])

        squares.append(square)

    return squares

def get_polygon_features(
    image,
    polygons,
    feature_extraction_fn,
    pixel_size,
    geometry_col_name: str = "geometry",
):
    """For a GeoPandas DataFrame of polygons,
    convert them to an integer labelmap and extract
    features of an image using feature_extraction_fn.
    
    Polygon features will be extracted independently, which
    will allow for overlap of objects, if they exist
    """

    results = {"cell_id": []}

    for cell_id, polygon in polygons.iterrows():

        label = polygon_to_mask(
            polygon=polygon[geometry_col_name],
            crop_window_size=image.shape[0],
            pixel_size=pixel_size
        )

        if numpy.count_nonzero(label) > 0:
            results["cell_id"].append(cell_id)

            features = feature_extraction_fn(image=image, labels=label)

            results = combine_dicts(results, features)

    return results

def polygon_to_mask(
    polygon,
    crop_window_size,
    pixel_size: float = 0.2125,
) -> numpy.array:
    """
    Rasterize a single polygon into a binary mask
    """
    output_array = numpy.zeros((crop_window_size, crop_window_size), dtype=numpy.int32)

    exterior_coords = numpy.array(polygon.exterior.coords)
    exterior_coords = numpy.multiply(exterior_coords, pixel_size**-1)
    rr, cc = skimage.draw.polygon(
        exterior_coords[:, 1], exterior_coords[:, 0], shape=output_array.shape
    )
    output_array[rr, cc] = True

    return output_array

def partial_within(a: shapely.Polygon, b: shapely.Polygon) -> bool:
    """Determine if a polygon partially intersects another.

    Args:
        a (shapely.Polygon): Parent polygon
        b (shapely.Polygon): Child polygon

    Returns:
        bool: True if polygons intersect, otherwise False.
    """
    polygon_intersection = a.intersection(b).area
    return polygon_intersection > 0

def validate_geodataframe(geo_dataframe: geopandas.GeoDataFrame, geometry_column: str = "geometry") -> geopandas.GeoDataFrame:
    """Create a subset of a GeoDataFrame that only has valid geometries

    This allows for avoiding of TopologyException errors, which are the result
    of malformed polygons.

    Args:
        geo_dataframe (geopandas.GeoDataFrame): GeoDataFrame to be validated
        geometry_column (str, optional): Column containing geometries. Defaults to "geometry".  

    Returns:
        geopandas.GeoDataFrame: Valid GeoDataFrame
    """
    geo_dataframe = geo_dataframe[geo_dataframe[geometry_column].is_valid]
    return geo_dataframe

def scale_spatialdata_polygons_and_add_cluster_id(
        sdata_path: str,
        cluster_id_csv_path: str,
        pixel_size: int,
        scale_level: int,
        polygon_key: str = "cell_boundaries",
) -> geopandas.GeoDataFrame:
    """Read a Xenium experiment as a SpatialData object and read the desired polygon information
    for rasterization. Additionally, also read in the polygons ID, which could be the result from leiden
    clustering. Requires cluster_id_csv to have a "cell_id" column for merging with the polygon cell_id

    Args:
        sdata_path (str): Path to Xenium experiment directory
        cluster_id_csv_path (str): Path to file containing cell_id and the associated cluster ID
        pixel_size (int): Pixel size of Xenium experiment
        scale_level (int): Scale level to transform the polygons to
        polygon_key (str, optional): Which polygons to load. Defaults to "cell_boundaries".

    Returns:
        geopandas.GeoDataFrame: GeoPandas DataFrame containing pixel scaled polygons with cell cluster ID 
        information stored in the "cell_type" column.
    """
    
    sdata = spatialdata_io.xenium(
        path=sdata_path,
        n_jobs=-1,
        # Only load the data that we need: boundary polygons
        cells_as_circles = False,
        cells_labels = False,
        nucleus_labels = False,
        transcripts = False,
        morphology_mip = False,
        morphology_focus = False,
        aligned_images = False,
        cells_table = False,
        nucleus_boundaries = (polygon_key == "nucleus_boundaries"),
        cells_boundaries = (polygon_key == "cell_boundaries")
    )

    cell_clusters = pandas.read_csv(cluster_id_csv_path)

    # Check that cell_clusters is as expected
    assert len(cell_clusters.columns) == 2, f"Expected 2 columns, got {len(cell_clusters.columns)}."
    assert "cell_id" in cell_clusters.columns, "Expected cell_id column in cell_clusters."
    
    # Set cell_id as the index for join
    cell_clusters = cell_clusters.set_index("cell_id")
    # Get the cluster ID column
    cluster_column = cell_clusters.columns[0]

    assert cell_clusters[cluster_column].min() > 0, f"Column {cluster_column} cannot be 0-indexed. 0 is reserved for the background value in semantic segmentation. Add 1 to {cluster_column} values to make them 1-indexed."

    # Rename cluster column to a fixed ID
    cell_clusters = cell_clusters.rename(columns={cluster_column: "cell_type"})

    geo_dataframe = sdata.shapes[polygon_key].join(cell_clusters)

    # Determine the required scaling
    scale_factor = ((pixel_size**-1) / 2**scale_level)

    # Apply scaling as an affine transform to all geometries
    geo_dataframe["geometry"] = geo_dataframe["geometry"].apply(
    lambda x: shapely.affinity.affine_transform(
        x, 
        [scale_factor, 0, 0, scale_factor, 0, 0]
        )
    )

    geo_dataframe = validate_geodataframe(geo_dataframe)

    return geo_dataframe

def common_index(
    df_list: typing.List[geopandas.GeoDataFrame], 
    ) -> list: 
    """Get the common indices from a list of DataFrames.

    Useful after having run validate_geodataframe on separate
    dataframes, prior to performing an polygon intersection. eg. 
    when finding the cytoplasm from nuclei and cell polygons. You've 
    got to ensure that all polygons in all groups are valid first and are
    in a pair.

    Args:
        df_list (typing.List[geopandas.GeoDataFrame]): List of DataFrames

    Returns:
        list: A list of index values common to all dataframes
    """
    assert len(df_list) > 1, "Can't find the intersection of a single DataFrame"
    index_sets = [set(df.index) for df in df_list]

    base_set = index_sets.pop(0)
    index_intesection = base_set.intersection(*index_sets)

    return list(index_intesection)

def get_cytoplasm_polygon(cell_boundaries: geopandas.GeoDataFrame, nucleus_boundaries: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Calculate cytoplasm polygons from cell and nuclei polygons.

    Also pre-filters GeoDataFrames to contain valid polygons.

    Args:
        cell_boundaries (geopandas.GeoDataFrame): Cell polygons
        nucleus_boundaries (geopandas.GeoDataFrame): Nuclei polygons

    Returns:
        geopandas.GeoDataFrame: Cytoplasm polygons
    """
    cell_boundaries = validate_geodataframe(cell_boundaries)
    nucleus_boundaries = validate_geodataframe(nucleus_boundaries)

    idx = common_index(
        [cell_boundaries, nucleus_boundaries]
    )

    cell_boundaries = cell_boundaries[cell_boundaries.index.isin(idx)]
    nucleus_boundaries = nucleus_boundaries[nucleus_boundaries.index.isin(idx)]

    cytoplasm_polygons = cell_boundaries.apply(
        lambda x: x.difference(nucleus_boundaries)
    )

    return cytoplasm_polygons

def create_geopandas_crops(
    geo_dataframe: geopandas.GeoDataFrame,
    top_and_left_idx: typing.List[typing.Tuple[int, int]],
    crop_window_size: typing.Tuple[int, int],
) -> typing.Dict[typing.Tuple[int, int], geopandas.GeoDataFrame]:
    """Generate a dictionary of cropped views of a dataframe. The key
    will represent the crop idx in the format (top, left) and the value
    will contain the polygons from geo_dataframe that were in this
    crop region.

    This dictionary can be used for multiprocess rasterization.

    Args:
        geo_dataframe (geopandas.GeoDataFrame): GeoDataFrame to be cropped
        crop_idx_list (typing.List[typing.Tuple[int, int]]): List of tuples representing (top, left) crop coodindates
        crop_window_size (typing.Tuple[int, int]): Desired crop window size

    Returns:
        typing.Dict[typing.Tuple[int, int], geopandas.GeoDataFrame]: Dict of (top, left) indices: cropped_polygons
    """
    top, left = top_and_left_idx
    bottom, right = top + crop_window_size[0], left + crop_window_size[1]

    # Define the cropping region as a polygon
    bbox = shapely.geometry.box(left, bottom, right, top)

    # Use sindex to quickly find possible matches
    # These will later be refined by partial_within
    possible_matches_index = list(geo_dataframe.sindex.intersection(bbox.bounds))
    geo_dataframe_subset = geo_dataframe.iloc[possible_matches_index]

    # Subset the DataFrame to be within the crop indices
    geo_dataframe_subset = geo_dataframe_subset[geo_dataframe_subset["geometry"].apply(
        lambda geom: partial_within(
            geom,
            bbox
        )
    # If empty, ensure the columns are still there
    )].reindex(columns=geo_dataframe_subset.columns)

    # If geo_dataframe_subset is empty, it leads to creation of 
    # if not isinstance(geo_dataframe_subset, geopandas.DataFrame):
        # geo_dataframe_subset = geopandas.DataFrame(geo_dataframe_subset)

    output_gdf = {}
    output_gdf[top, left] = geo_dataframe_subset
    return output_gdf

def create_geopandas_crops3(
    shared_geo_dataframe_name: geopandas.GeoDataFrame,
    wkb_polygon_array_shape: typing.Tuple[int, int],
    wkb_polygon_array_datatype: numpy.dtypes.ObjectDType,
    top_and_left_idx: typing.List[typing.Tuple[int, int]],
    crop_window_size: typing.Tuple[int, int],
) -> typing.Dict[typing.Tuple[int, int], geopandas.GeoDataFrame]:
    """Generate a dictionary of cropped views of a dataframe. The key
    will represent the crop idx in the format (top, left) and the value
    will contain the polygons from geo_dataframe that were in this
    crop region.

    This dictionary can be used for multiprocess rasterization.

    Args:
        geo_dataframe (geopandas.GeoDataFrame): GeoDataFrame to be cropped
        crop_idx_list (typing.List[typing.Tuple[int, int]]): List of tuples representing (top, left) crop coodindates
        crop_window_size (typing.Tuple[int, int]): Desired crop window size

    Returns:
        typing.Dict[typing.Tuple[int, int], geopandas.GeoDataFrame]: Dict of (top, left) indices: cropped_polygons
    """

    # Access shared memory
    existing_shm = SharedMemory(name=shared_mem_name)
    wkb_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Convert WKB back to geometries
    geometries = [shapely.wkb.loads(wkb) for wkb in wkb_array]

    # Create a GeoDataFrame from the geometries
    geo_dataframe = gpd.GeoDataFrame({'geometry': geometries})

    

    top, left = top_and_left_idx
    bottom, right = top + crop_window_size[0], left + crop_window_size[1]

    # Define the cropping region as a polygon
    bbox = shapely.geometry.box(left, bottom, right, top)

    # Use sindex to quickly find possible matches
    # These will later be refined by partial_within
    possible_matches_index = list(geo_dataframe.sindex.intersection(bbox.bounds))
    geo_dataframe_subset = geo_dataframe.iloc[possible_matches_index]

    # Subset the DataFrame to be within the crop indices
    geo_dataframe_subset = geo_dataframe_subset[geo_dataframe_subset["geometry"].apply(
        lambda geom: partial_within(
            geom,
            bbox
        )
    # If empty, ensure the columns are still there
    )].reindex(columns=geo_dataframe_subset.columns)

    # If geo_dataframe_subset is empty, it leads to creation of 
    # if not isinstance(geo_dataframe_subset, geopandas.DataFrame):
        # geo_dataframe_subset = geopandas.DataFrame(geo_dataframe_subset)

    output_gdf = {}
    output_gdf[top, left] = geo_dataframe_subset
    return output_gdf


def sjoin_with_overlap_threshold(left_gdf, right_gdf, overlap_threshold=0.5, 
                                overlap_method='left', how='inner', predicate='intersects'):
    """
    Perform a spatial join between two GeoDataFrames, keeping only pairs
    where polygons overlap by at least the specified percentage.
    
    Parameters:
    -----------
    left_gdf : GeoDataFrame
        Left GeoDataFrame (typically the "target" geometries)
    right_gdf : GeoDataFrame  
        Right GeoDataFrame (typically the "source" geometries)
    overlap_threshold : float, default 0.5
        Minimum overlap percentage (0.0 to 1.0). E.g., 0.5 means 50% overlap required
    overlap_method : str, default 'left'
        How to calculate overlap percentage:
        - 'left': overlap area / left geometry area
        - 'right': overlap area / right geometry area  
        - 'smaller': overlap area / area of smaller geometry
        - 'larger': overlap area / area of larger geometry
        - 'average': overlap area / average of both areas
    how : str, default 'inner'
        Type of join ('left', 'right', 'inner')
    predicate : str, default 'intersects'
        Spatial predicate for initial join
        
    Returns:
    --------
    GeoDataFrame
        Joined GeoDataFrame with only overlaps meeting the threshold
    """
    
    # Ensure geometries are valid
    left_gdf = left_gdf.copy()
    right_gdf = right_gdf.copy()
    
    # Initial spatial join to find intersecting pairs
    joined = geopandas.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
    
    if joined.empty:
        return joined
    
    # Calculate overlap areas and percentages
    overlaps = []
    
    for idx, row in joined.iterrows():
        # Get the geometries
        left_geom = row.geometry
        right_idx = row['index_right']
        right_geom = right_gdf.loc[right_idx, 'geometry']
        
        # Calculate intersection area
        try:
            intersection = left_geom.intersection(right_geom)
            overlap_area = intersection.area
            
            if overlap_area == 0:
                overlap_pct = 0
            else:
                # Calculate areas
                left_area = left_geom.area
                right_area = right_geom.area
                
                # Calculate overlap percentage based on method
                if overlap_method == 'left':
                    overlap_pct = overlap_area / left_area if left_area > 0 else 0
                elif overlap_method == 'right':
                    overlap_pct = overlap_area / right_area if right_area > 0 else 0
                elif overlap_method == 'smaller':
                    min_area = min(left_area, right_area)
                    overlap_pct = overlap_area / min_area if min_area > 0 else 0
                elif overlap_method == 'larger':
                    max_area = max(left_area, right_area)
                    overlap_pct = overlap_area / max_area if max_area > 0 else 0
                elif overlap_method == 'average':
                    avg_area = (left_area + right_area) / 2
                    overlap_pct = overlap_area / avg_area if avg_area > 0 else 0
                else:
                    raise ValueError(f"Invalid overlap_method: {overlap_method}")
            
            overlaps.append({
                'index': idx,
                'overlap_area': overlap_area,
                'overlap_percentage': overlap_pct,
                'left_area': left_area,
                'right_area': right_area
            })
            
        except Exception as e:
            # Handle invalid geometries
            print(f"Warning: Could not calculate overlap for index {idx}: {e}")
            overlaps.append({
                'index': idx,
                'overlap_area': 0,
                'overlap_percentage': 0,
                'left_area': 0,
                'right_area': 0
            })
    
    # Convert to DataFrame and merge with joined data
    overlap_df = pandas.DataFrame(overlaps).set_index('index')
    result = joined.join(overlap_df)
    
    # Filter by overlap threshold
    filtered_result = result[result['overlap_percentage'] >= overlap_threshold].copy()
    
    return filtered_result