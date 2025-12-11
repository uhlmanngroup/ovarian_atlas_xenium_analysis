
import logging
import os
from pathlib import Path

import fire
import numpy as np

import tifffile as tf
import yaml

from shapely.affinity import translate
from shapely.geometry import Polygon
from skimage.draw import polygon2mask


def get_roi_as_arrays(roi):
    roi_arrays = np.array(
        [
            [float(v) for v in roi_points.split(",") if v]
            for roi_points in roi.getPoints().val.split(" ")
        ]
    )
    return roi_arrays


def getChannelIndex(image, label):
    labels = image.getChannelLabels()
    if label in labels:
        idx = labels.index(label)
        print("Channel Index:", idx)
        return idx
    return 0


def hidden_input(fn):
    import getpass
    def wrapper(*args, **kwargs):
        if kwargs.get("omero_password") is None:
            kwargs["omero_password"] = getpass.getpass()
        return fn(*args, **kwargs)

    return wrapper


def get_ROI_image(poly, channels, tile_size=(512, 512)):
    height = poly.bounds[3] - poly.bounds[1]
    width = poly.bounds[2] - poly.bounds[0]
    print(poly.bounds, height, width)
    shifted_poly = translate(poly, -poly.bounds[0], -poly.bounds[1])
    mask = (
        polygon2mask((int(width), int(height)), list(shifted_poly.exterior.coords))
        .astype(np.uint8)
        .T
    )

    tilelist = []
    for c in range(len(channels)):
        for y in range(int(poly.bounds[1]), int(poly.bounds[3]), tile_size[1]):
            for x in range(int(poly.bounds[0]), int(poly.bounds[2]), tile_size[0]):
                tilelist.append((0, c, 0, (x, y, tile_size[0], tile_size[1])))
    return mask, tilelist


@hidden_input
def get_omero_roi(
    image_id,
    omero_id,
    save_dir,
    omero_password=None,
    # host="wsi-omero-prod-01.internal.sanger.ac.uk",
    host="wsi-omero-prod-02.internal.sanger.ac.uk",
    download_image=False,
    local_crop=False,
):
    import omero.clients
    from omero.gateway import BlitzGateway

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Conncet to Omero
    with BlitzGateway(
        omero_id, omero_password, host=host, secure=True, port=4064
    ) as conn:
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        group = conn.getGroupFromContext()
        print("Current group: ", group.getName())
        # Find the image across all grouos
        image = conn.getObject("Image", image_id)
        print("Image: ", image)
        if image is not None:
            print(f"Group: {image.getDetails().getGroup().getName()}")
            print(
                image.getDetails().getGroup().getId()
            )  # access groupId without loading group
        group_id = image.getDetails().getGroup().getId()
        # This is how we 'switch group' in webclient
        conn.SERVICE_OPTS.setOmeroGroup(group_id)
        conn.setGroupForSession(group_id)
        group = conn.getGroupFromContext()
        print("Current group: ", group.getName())
        projects = conn.listProjects()
        for p in projects:
            print(p.getName(), "Owner: ", p.getDetails().getOwner().getFullName())
        image = conn.getObject("Image", image_id)
        path = image.getImportedImageFilePaths()
        path = path["client_paths"]
        assert len(path) == 1
        path = "/" + path[0]
        logging.info(path)
        roi_service = conn.getRoiService()
        result = roi_service.findByImage(image_id, None)
        roi_ids = [roi.id.val for roi in result.rois]
        print(roi_ids)
        group = conn.getGroupFromContext()

        pixels = image.getPrimaryPixels()
        channels = image.getChannels()
        labels = image.getChannelLabels()
        print(channels, labels)

        print("Current group: ", group.getName())
        # Get the ROIs for each image
        for roi in result.rois:
            roi_id = roi.getId().getValue()
            print(f"ROI:  ID: {roi_id}")
            for s in roi.copyShapes():
                roi_name = None
                if s.getTextValue():
                    roi_name = s.getTextValue().getValue()
                if isinstance(s, omero.model.PolygonI):
                    print("loading polygon")
                    poly = Polygon(get_roi_as_arrays(s))
                    height = poly.bounds[3] - poly.bounds[1] - 1
                    width = poly.bounds[2] - poly.bounds[0] - 1
                    if height == 0 or width == 0:
                        continue
                    print(poly.bounds, height, width)
                    bbox_str = "_".join([str(int(v)) for v in poly.bounds])
                    for_bfconvert_crop = f"{int(poly.bounds[0])},{int(poly.bounds[1])},{int(width)},{int(height)}"
                    print("polygon loaded")
                elif isinstance(s, omero.model.RectangleI):
                    print("loading rectangle")
                    height = s.height.val - 1
                    width = s.width.val - 1
                    if height == 0 or width == 0:
                        continue
                    poly = Polygon(
                        [
                            [s.x.val, s.y.val],
                            [s.x.val + width, s.y.val],
                            [s.x.val + width, s.y.val + height],
                            [s.x.val, s.y.val + height],
                        ],
                    )
                    bbox_str = "_".join([str(int(v)) for v in poly.bounds])
                    for_bfconvert_crop = f"{int(poly.bounds[0])},{int(poly.bounds[1])},{int(width)},{int(height)}"
                    print("rectangle loaded")
                else:
                    print("This type of ROI is not supported")
                    continue
                out_prefix = f"{roi_id}_{roi_name}_{bbox_str}".replace("/", "-")

                # Download crops
                if download_image:
                    mask, tilelist = get_ROI_image(poly, channels)
                    tf.imwrite(
                        f"{save_dir}/{out_prefix}.tif",
                        pixels.getTiles(tilelist),
                        tile=(512, 512),
                        shape=(len(channels), int(height), int(width)),
                        dtype=np.dtype("uint16"),
                        # bigtiff=True,
                    )
                    print("image saved")
                    tf.imwrite(
                        f"{save_dir}/{out_prefix}_mask.tif",
                        mask,
                        dtype=np.dtype("uint8"),
                        shape=mask.shape,
                    )
                    print("mask saved")
                # logging.info(poly, labels, path)
                infos = {
                    "poly": poly.wkt,
                    "channel_labels": labels,
                    "raw_image_path": path,
                    "for_bfconvert_crop": for_bfconvert_crop,
                    "roi_name": roi_name,
                    "roi_id": roi_id,
                }
                with open(f"{save_dir}/{out_prefix}.yaml", "w") as file:
                    yaml.dump(infos, file)
                if local_crop:
                    stem = Path(path).stem
                    with open(f"{save_dir}/{out_prefix}_crop.sh", "w") as file:
                        file.write(
                            f"bfconvert -pyramid-resolutions 10 -crop {for_bfconvert_crop} -series 0 -bigtiff '{path}' '{stem}_{out_prefix}.ome.tif'"
                        )
