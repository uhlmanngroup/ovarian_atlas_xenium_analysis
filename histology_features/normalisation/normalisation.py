import typing
import numpy
from scipy.interpolate import Akima1DInterpolator
import skimage

def collect_img_stats(img_list, norm_percentiles=[1, 5, 95, 99], mask_list=None):
    """
    Adapted from VALIS.

    Collect image statistics such as histogram and percentiles.
    
    Parameters:
        img_list (list of numpy.ndarray): List of 2D or 3D image arrays to process.
        norm_percentiles (list of float): Percentiles to compute for normalization.
        mask_list (list of numpy.ndarray or None, optional): 
            List of masks corresponding to the images. If None, no masking is applied.

    Returns:
        tuple:
            - all_histogram (numpy.ndarray): Combined histogram of all images (256 bins).
            - all_img_stats (numpy.ndarray): Array containing percentile values and mean.
    """
    # Determine if masks are being used
    use_masks = mask_list is not None and any(mask is not None for mask in mask_list)

    # Initialize combined histogram and statistics variables
    img0 = img_list[0][mask_list[0] > 0] if use_masks else img_list[0].ravel()
    all_histogram, _ = numpy.histogram(img0, bins=256)
    total_pixels = img0.size
    total_sum = img0.sum()

    # Process remaining images
    for i in range(1, len(img_list)):
        img = img_list[i]
        if use_masks and mask_list[i] is not None:
            img_flat = img[mask_list[i] > 0]
        else:
            img_flat = img.ravel()

        # Update histogram and cumulative statistics
        img_hist, _ = numpy.histogram(img_flat, bins=256)
        all_histogram += img_hist
        total_pixels += img_flat.size
        total_sum += img_flat.sum()

    # Compute mean
    mean_value = total_sum / total_pixels

    # Calculate percentiles from the cumulative distribution function (CDF)
    ref_cdf = 100 * numpy.cumsum(all_histogram) / total_pixels
    percentile_values = [numpy.searchsorted(ref_cdf, p, side='left') for p in norm_percentiles]

    # Combine percentiles and mean into final statistics array
    all_img_stats = numpy.array(percentile_values + [mean_value])

    return all_histogram, all_img_stats


def norm_img_stats(img, target_stats):
    """
    From VALIS.

    Normalize an image

    Image will be normalized to have same stats as target_stats

    Based on method in
    "A nonlinear mapping approach to stain normalization in digital histopathology
    images using image-specific color deconvolution.", Khan et al. 2014

    Assumes that img values range between 0-255
    """

    _, src_stats_flat = collect_img_stats([img])

    # Avoid duplicates and keep in ascending order
    lower_knots = numpy.array([0])
    upper_knots = numpy.array([300, 350, 400, 450])
    src_stats_flat = numpy.hstack([lower_knots, src_stats_flat, upper_knots]).astype(float)
    target_stats_flat = numpy.hstack([lower_knots, target_stats, upper_knots]).astype(float)

    # Add epsilon to avoid duplicate values
    eps = 100*numpy.finfo(float).resolution
    eps_array = numpy.arange(len(src_stats_flat)) * eps
    src_stats_flat = src_stats_flat + eps_array
    target_stats_flat = target_stats_flat + eps_array

    # Make sure src stats are in ascending order
    src_order = numpy.argsort(src_stats_flat)
    src_stats_flat = src_stats_flat[src_order]
    target_stats_flat = target_stats_flat[src_order]

    cs = Akima1DInterpolator(src_stats_flat, target_stats_flat)

    normed_img = cs(img.reshape(-1)).reshape(img.shape)

    if img.dtype == numpy.uint8:
        normed_img = numpy.clip(normed_img, 0, 255)

    return normed_img


def normalise_sequential_images(
    image_list: typing.List[numpy.ndarray]
) -> typing.List[numpy.ndarray]:
    """
    Normalise a sequence of images so they are more similar
    to one another.
    """

    _, all_img_stats = collect_img_stats(image_list)

    norm_images = []

    for i in image_list:
        norm_img = norm_img_stats(i, all_img_stats)
        norm_img = skimage.exposure.rescale_intensity(norm_img, out_range=(0, 255)).astype(numpy.uint8)
        norm_images.append(norm_img)

    return norm_images
