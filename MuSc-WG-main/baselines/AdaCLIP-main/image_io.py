import os

import cv2
import numpy as np


def read_image_cv2(path, flags=cv2.IMREAD_COLOR):
    image_path = os.fspath(path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = None
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size > 0:
            image = cv2.imdecode(data, flags)
    except OSError:
        image = None

    if image is None or image.size == 0:
        image = cv2.imread(image_path, flags)

    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    return image


def write_image_cv2(path, image):
    image_path = os.fspath(path)
    parent = os.path.dirname(image_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(image_path)[1]
    if not ext:
        raise ValueError(f"Output image path has no extension: {image_path}")

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"OpenCV could not encode image for: {image_path}")

    encoded.tofile(image_path)
    return True
