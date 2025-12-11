import io
import os
import matplotlib.pyplot as plt
import skimage
import imageio
import numpy
from contextlib import contextmanager


def make_gif(filename: str, array_list: list, **kwargs):
    """
    Create a gif from a list of arrays
    """
    assert filename.endswith(".gif"), """Filename must end with .gif"""

    with imageio.get_writer(filename, mode="I", loop=0, **kwargs) as writer:
        for i in array_list:
            writer.append_data(i)


def fig_to_numpy(figure_object):
    io_buf = io.BytesIO()
    figure_object.savefig(io_buf, format="raw")
    io_buf.seek(0)
    img_arr = numpy.reshape(
        numpy.frombuffer(io_buf.getvalue(), dtype=numpy.uint8),
        newshape=(
            int(figure_object.bbox.bounds[3]),
            int(figure_object.bbox.bounds[2]),
            -1,
        ),
    )
    io_buf.close()
    return img_arr

def sort_by_step(list_of_wandb_image_paths: list):
    return sorted(
        list_of_wandb_image_paths, 
        key=lambda x: int(x.split("_")[-2])
        )

def gather_files(left_path, right_path):
    for root, dirs, files in os.walk(left_path):
        left_files = [os.path.join(root, file) for file in files]
    for root, dirs, files in os.walk(right_path):
        right_files = [os.path.join(root, file) for file in files]

    left_files = sort_by_step(left_files)
    right_files = sort_by_step(right_files)

    return left_files, right_files

@contextmanager
def suppress_rendering():
    plt.ioff()  # Turn off interactive mode (prevents rendering)
    try:
        yield
    finally:
        plt.ion() 

def left_right_plot(left, right, minmax_scale: bool = False):
    assert len(left) == len(right)

    output = []

    for i in range(len(left)):
        step_id = int(left[i].split("_")[-2])
        left_img = skimage.io.imread(left[i])
        right_img = skimage.io.imread(right[i])
        if minmax_scale:
            left_img = (left_img - left_img.min()) / (left_img.max() - left_img.min())
            right_img = (right_img - right_img.min()) / (right_img.max() - right_img.min())
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(left_img)
        ax[1].imshow(right_img)

        ax[0].axis("off")
        ax[1].axis("off")

        ax[1].set_title(f"Training step {step_id:,}")

        output.append(fig_to_numpy(fig))

    return output
