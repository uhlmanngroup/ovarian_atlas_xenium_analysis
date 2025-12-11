import pathlib
import fnmatch
import typing

def isolate_sdata_segmentations(
    sdata: "SpatialData",
    save_path: str,
    segmentation_key: str,
    scaling_factor: typing.Tuple[float, float, float] = (1, 1, 1),
):
    """
    Save segmentations stored in a SpatialData object
    to disk. 

    Useful if segmentations are to be used in another tool.
    """

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "*"  in segmentation_key:
        segmentation_key = wildcard_search(
            sdata.shapes, segmentation_key
        )

        assert len(segmentation_key) != 0, f"No matches found for {segmentation_key}"

    segmentations = sdata[segmentation_key]

    segmentations = segmentations.scale(
        *scaling_factor, 
        origin=(0, 0)
    )

    segmentations.to_file(
        save_path, 
        driver='GeoJSON'
    )

def wildcard_search(keys, pattern):
    return fnmatch.filter(keys, pattern)