import cv2
import numpy as np


def extract_pixel_features(image):

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(
        np.float32
    ) / 255.0


    blur3 = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    blur5 = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    blur9 = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )


    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )


    gx = gx / 8.0
    gy = gy / 8.0


    magnitude = cv2.magnitude(
        gx,
        gy
    )


    local_mean5 = cv2.blur(
        gray,
        (5, 5)
    )

    local_mean9 = cv2.blur(
        gray,
        (9, 9)
    )


    gray_squared = gray * gray

    mean_squared5 = cv2.blur(
        gray_squared,
        (5, 5)
    )


    variance5 = (
        mean_squared5
        -
        local_mean5 * local_mean5
    )

    variance5 = np.maximum(
        variance5,
        0
    )

    local_std5 = np.sqrt(
        variance5
    )


    local_contrast = (
        gray - local_mean9
    )


    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )


    yy, xx = np.mgrid[
        0:128,
        0:128
    ]


    xx = xx.astype(
        np.float32
    ) / 127.0

    yy = yy.astype(
        np.float32
    ) / 127.0


    distance_center = np.sqrt(
        (xx - 0.5) ** 2
        +
        (yy - 0.5) ** 2
    )


    features = np.stack(
        [
            gray,
            blur3,
            blur5,
            blur9,
            gx,
            gy,
            magnitude,
            local_mean5,
            local_mean9,
            local_std5,
            local_contrast,
            laplacian,
            xx,
            yy,
            distance_center
        ],
        axis=2
    )


    features = features.reshape(
        -1,
        features.shape[2]
    )


    return features.astype(
        np.float32
    )

