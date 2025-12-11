import typing

import numpy


def apply_pixel_classifier(
    image: numpy.ndarray, classifier: typing.Callable, class_mappings: dict = None
) -> numpy.ndarray:
    features = image.squeeze()
    prediction = classifier([features])
    if class_mappings:
        # Inverse the dict for easy slicing
        inversed = dict(zip(class_mappings.values(), class_mappings.keys()))
        prediction = inversed[prediction[0]]
    return numpy.array(prediction)
