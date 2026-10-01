import os
import cv2
import joblib
import numpy as np
from config import (
    MODEL_FILE,
    CLASSIFICATION_MODEL_FILE,
    SETTINGS_FILE
)

from services.feature_extraction import (
    extract_pixel_features
)

from services.preprocessing import (
    preprocess_image
)

from services.postprocessing import (
    post_process_mask
)
from services.classification_service import (
    classify_image
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        "Model file not found: "
        + MODEL_FILE
    )


model = joblib.load(
    MODEL_FILE
)

# LOAD CLASSIFICATION MODEL
if not os.path.exists(CLASSIFICATION_MODEL_FILE):
    raise FileNotFoundError(
        "Classification model file not found: "
        + CLASSIFICATION_MODEL_FILE
    )

classification_model = joblib.load(
    CLASSIFICATION_MODEL_FILE
)


# ============================================================
# LOAD SETTINGS
# ============================================================

if os.path.exists(SETTINGS_FILE):

    settings = joblib.load(
        SETTINGS_FILE
    )

else:

    settings = {}


PREDICTION_THRESHOLD = settings.get(
    "prediction_threshold",
    0.50
)


# ============================================================
# SEGMENT IMAGE
# ============================================================
# ============================================================
# SEGMENT IMAGE
# ============================================================

def segment_image(image):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image = preprocess_image(
        image
    )


    # --------------------------------------------------------
    # CLASSIFICATION FIRST
    # --------------------------------------------------------

    tumor_type = classify_image(
        image,
        classification_model
    )


    # --------------------------------------------------------
    # NO TUMOR
    # --------------------------------------------------------

    if str(tumor_type).strip().lower() == "no_tumor":

        # Create completely empty mask
        mask = np.zeros(
            (128, 128),
            dtype=np.uint8
        )

        tumor_area_percent = 0.0

        return (
            mask,
            tumor_area_percent,
            "no_tumor"
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = extract_pixel_features(
        image
    )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )


    tumor_probability = probabilities[
        :,
        1
    ]


    tumor_probability = tumor_probability.reshape(
        128,
        128
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    mask = np.where(
        tumor_probability >= PREDICTION_THRESHOLD,
        255,
        0
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # POST PROCESS
    # --------------------------------------------------------

    mask = post_process_mask(
        mask
    )


    # --------------------------------------------------------
    # TUMOR AREA
    # --------------------------------------------------------

    tumor_pixels = np.sum(
        mask > 0
    )

    total_pixels = (
        mask.shape[0]
        *
        mask.shape[1]
    )

    tumor_area_percent = (
        tumor_pixels
        /
        total_pixels
    ) * 100


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return (
        mask,
        tumor_area_percent,
        tumor_type
    )


# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    mask
):

    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )


    overlay = image.copy()


    tumor = mask > 0


    # Red tumor region
    overlay[tumor] = (
        0,
        0,
        255
    )


    result = cv2.addWeighted(
        image,
        0.70,
        overlay,
        0.30,
        0
    )


    return result
