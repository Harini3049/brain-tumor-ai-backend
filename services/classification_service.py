import cv2
import numpy as np

from config import IMG_SIZE


CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


def calculate_hog(image):

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

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

    magnitude, angle = cv2.cartToPolar(
        gx,
        gy,
        angleInDegrees=True
    )

    cell_size = 8
    block_size = 2
    bins = 9

    angle = angle % 180

    cells_y = IMG_SIZE // cell_size
    cells_x = IMG_SIZE // cell_size

    cell_histograms = np.zeros(
        (
            cells_y,
            cells_x,
            bins
        ),
        dtype=np.float32
    )

    for cy in range(cells_y):

        for cx in range(cells_x):

            y_start = cy * cell_size
            y_end = y_start + cell_size

            x_start = cx * cell_size
            x_end = x_start + cell_size

            cell_angles = angle[
                y_start:y_end,
                x_start:x_end
            ]

            cell_magnitude = magnitude[
                y_start:y_end,
                x_start:x_end
            ]

            cell_angles = cell_angles.flatten()
            cell_magnitude = cell_magnitude.flatten()

            histogram = np.zeros(
                bins,
                dtype=np.float32
            )

            for a, m in zip(
                cell_angles,
                cell_magnitude
            ):

                bin_number = int(
                    a // 20
                )

                if bin_number >= bins:
                    bin_number = bins - 1

                histogram[
                    bin_number
                ] += m

            cell_histograms[
                cy,
                cx
            ] = histogram

    hog_features = []

    for y in range(
        cells_y - block_size + 1
    ):

        for x in range(
            cells_x - block_size + 1
        ):

            block = cell_histograms[
                y:y + block_size,
                x:x + block_size
            ]

            block = block.flatten()

            norm = np.sqrt(
                np.sum(
                    block ** 2
                ) + 1e-6
            )

            block = block / norm

            hog_features.extend(
                block
            )

    return np.array(
        hog_features,
        dtype=np.float32
    )


def calculate_color_histogram(image):

    features = []

    for channel in range(3):

        histogram = cv2.calcHist(
            [image],
            [channel],
            None,
            [32],
            [0, 256]
        )

        histogram = cv2.normalize(
            histogram,
            histogram
        )

        features.extend(
            histogram.flatten()
        )

    return np.array(
        features,
        dtype=np.float32
    )


def extract_classification_features(image):

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    hog_features = calculate_hog(
        image
    )

    color_features = calculate_color_histogram(
        image
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(
        np.float32
    ) / 255.0

    mean_value = np.mean(gray)
    std_value = np.std(gray)
    min_value = np.min(gray)
    max_value = np.max(gray)

    intensity_features = np.array(
        [
            mean_value,
            std_value,
            min_value,
            max_value
        ],
        dtype=np.float32
    )

    features = np.concatenate(
        [
            hog_features,
            color_features,
            intensity_features
        ]
    )

    return features.astype(
        np.float32
    )


def classify_image(
    image,
    classification_model
):

    features = extract_classification_features(
        image
    )

    features = features.reshape(
        1,
        -1
    )

    prediction = classification_model.predict(
        features
    )

    predicted_class = CLASSES[
        int(prediction[0])
    ]

    return predicted_class