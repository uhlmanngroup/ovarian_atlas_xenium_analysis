import spatialdata
import spatialdata_plot
import matplotlib.pyplot as plt
from histology_features.spec.xenium import Xenium
import numpy
from shapely.affinity import translate
from rasterio.features import rasterize

def validate_registration(
    sdata,
    target_image_key: str,
    aligned_image_key: str,
    target_shapes_key: str,
    aligned_shapes_key: str,
    scale_microns: bool = True,
    num_crops: int = 2,
):
    """
    Create plots for a spatialdata object to validate the alignment
    """

    target_shapes = sdata.shapes[target_shapes_key]
    aligned_shapes = sdata.shapes[aligned_shapes_key]

    if scale_microns:
        target_shapes = target_shapes.scale(
            Xenium.pixel_size.value, Xenium.pixel_size.value, origin=(0, 0)
        )
        aligned_shapes = aligned_shapes.scale(
            Xenium.pixel_size.value, Xenium.pixel_size.value, origin=(0, 0)
        )

    target_query_bounds = find_crops(
        # geo_df=target_shapes,
        geo_df=aligned_shapes,
        num_crops=num_crops
    )

    fig, ax = plt.subplots(num_crops, 2)

    for i, bounds in enumerate(target_query_bounds):
        print(bounds)
        crop_sdata = spatialdata.bounding_box_query(
            sdata, 
            axes=("x", "y"), 
            min_coordinate=[bounds[0], bounds[1]], 
            max_coordinate=[bounds[2], bounds[3]], 
            target_coordinate_system="global"
        )

        print(crop_sdata)

        # crop_sdata.pl.render_images(
        #     target_image_key
        # ).pl.render_shapes(
        #     target_shapes_key,
        #     fill_alpha=0.5,
        #     color="green",
        # ).pl.show(ax=ax[0, i], title="Target image - Aligned shapes")

        # crop_sdata.pl.render_images(
        #     aligned_image_key
        # ).pl.render_shapes(
        #     target_shapes_key,
        #     fill_alpha=0.5,
        #     color="green",
        # ).pl.show(ax=ax[0, i], title="Target image - Aligned shapes")

        crop_sdata.pl.render_images(
            target_image_key
        ).pl.show(ax=ax[i, 0], title="Target image - Aligned shapes")

        crop_sdata.pl.render_images(
            aligned_image_key
        ).pl.show(ax=ax[i, 1], title="Target image - Aligned shapes")

        # crop_sdata.pl.render_images(
        #     aligned_image_key
        # ).pl.render_shapes(
        #     target_shapes_key,
        #     fill_alpha=0.5,
        #     color="green",
        # ).pl.show(ax=ax[1, i], title="Aligned image - Target shapes")

    return fig


def find_crops(
    geo_df,
    num_crops,
    bounds_multiplier: float = 5,
):
    """For a given geopandas dataframe, find YX cropping
    coordinates around a subsample of the shapes.

    Returns array with shape (num_crops, 4)
    """

    return (geo_df.sample(num_crops).bounds * bounds_multiplier).to_numpy()


def plot_per_polygon_crops(gdf, img, polygon_indices=None, padding=10):
    """
    For each polygon (in pixel coords), crop the image with padding,
    rasterize the polygon mask, and plot the cropped image with overlay.

    Parameters:
        gdf (GeoDataFrame): Polygon geometries in pixel coordinates.
        img (numpy.ndarray): Image array of shape (C, H, W).
        polygon_indices (list[int] or None): Indices of polygons to plot. If None, all are used.
        padding (int): Padding (in pixels) around each crop.

    Returns:
        List of cropped image arrays (one per polygon).
    """
    assert img.ndim == 3, "Expected array shape (C, H, W)"
    C, H, W = img.shape

    if polygon_indices is None:
        # Plot 5 by default
        polygon_indices = numpy.random.randint(0, len(gdf), 5)

    cropped_images = []

    for idx in polygon_indices:
        polygon = gdf.geometry.iloc[idx]
        minx, miny, maxx, maxy = polygon.bounds

        # Compute padded bounds
        row_min = max(0, int(numpy.floor(miny) - padding))
        row_max = min(H, int(numpy.ceil(maxy) + padding))
        col_min = max(0, int(numpy.floor(minx) - padding))
        col_max = min(W, int(numpy.ceil(maxx) + padding))

        # Crop image
        cropped = img[:, row_min:row_max, col_min:col_max]
        cropped_images.append(cropped)

        # Shift polygon to cropped coordinate system
        shifted_polygon = translate(polygon, xoff=-col_min, yoff=-row_min)

        # Rasterize polygon
        mask = rasterize(
            [(shifted_polygon, 1)],
            out_shape=cropped.shape[1:],
            fill=0,
            dtype='uint8'
        )

        # Normalize RGB
        rgb = cropped[:3] if C >= 3 else numpy.repeat(cropped[0:1], 3, axis=0)
        rgb = numpy.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-5)

        # Plot cropped RGB with overlay
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.imshow(mask, cmap='Reds', alpha=0.4)
        plt.title(f"Cropped Polygon {idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return cropped_images