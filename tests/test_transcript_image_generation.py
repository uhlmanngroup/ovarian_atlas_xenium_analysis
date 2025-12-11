import pandas
import numpy
from histology_features.features.crop import dataframe_crop, get_crop_indices
import dask.dataframe

class TestTranscriptImageGeneration:
    def test_make_transcript_crops(self):
        df = pandas.DataFrame({
            "x_location": [0, 1, 2, 3],
            "y_location": [0, 1, 2, 3],
            "feature_name": ["a", "b", "c", "d"],
            "feature_name_idx": [0, 1, 2, 3],
            "qv": [10, 20, 30, 40]
        })

        label_encoding = {
            k:v for k, v in zip(df["feature_name"], df["feature_name_idx"])
        }

        df = dask.dataframe.from_pandas(df, npartitions=1)

        image_shape = (4, 4)

        crop_window_size = [2, 2]

        crops = [
            dataframe_crop(
                df=df,
                top=crop_idx[0],
                left=crop_idx[1],
                z_dimension_encoding=label_encoding,
                crop_window_size=crop_window_size,
            )
            for crop_idx in get_crop_indices(image_shape, crop_window_size)
        ]

        expected_crops = numpy.zeros((4, *crop_window_size, len(label_encoding)))
        expected_crops[0, 0, 0, 0] = 10
        expected_crops[0, 1, 1, 1] = 20
        expected_crops[3, 0, 0, 2] = 30
        expected_crops[3, 1, 1, 3] = 40

        numpy.testing.assert_array_almost_equal(crops, expected_crops)