import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MASK_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OVERLAY_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "brain_segmentation_random_forest_v3.pkl"
)
CLASSIFICATION_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "brain_tumor_svm_model.pkl"
)
SETTINGS_FILE = os.path.join(
    MODEL_DIR,
    "brain_segmentation_settings_v3.pkl"
)


IMG_SIZE = 128


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    MASK_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_OUTPUT_DIR,
    exist_ok=True
)
