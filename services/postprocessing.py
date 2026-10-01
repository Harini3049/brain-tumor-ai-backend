import cv2
import numpy as np


MIN_COMPONENT_SIZE = 20


def post_process_mask(mask):

    kernel_small = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )


    kernel_medium = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium
    )


    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )


    cleaned = np.zeros_like(
        mask
    )


    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_SIZE:

            cleaned[
                labels == i
            ] = 255


    return cleaned
