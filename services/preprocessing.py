import cv2


def preprocess_image(image):

    if image is None:
        raise ValueError(
            "Invalid image"
        )

    if len(image.shape) != 3:
        raise ValueError(
            "Please upload a color MRI image"
        )

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    return image
