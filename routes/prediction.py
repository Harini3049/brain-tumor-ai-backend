import os
import uuid
import cv2

from flask import (
    Blueprint,
    request,
    jsonify,
    send_file
)

from services.segmentation_service import (
    segment_image,
    create_overlay
)

from config import (
    MASK_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR
)


prediction_bp = Blueprint(
    "prediction",
    __name__
)


@prediction_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400


        file = request.files[
            "image"
        ]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = file.read()


        import numpy as np

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )


        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        mask, tumor_area ,tumor_type = segment_image(
            image
        )


        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            image,
            mask
        )


        # ----------------------------------------------------
        # Unique filename
        # ----------------------------------------------------

        result_id = uuid.uuid4().hex


        mask_filename = (
            result_id
            +
            "_mask.png"
        )


        overlay_filename = (
            result_id
            +
            "_overlay.png"
        )


        mask_path = os.path.join(
            MASK_OUTPUT_DIR,
            mask_filename
        )


        overlay_path = os.path.join(
            OVERLAY_OUTPUT_DIR,
            overlay_filename
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        cv2.imwrite(
            mask_path,
            mask
        )


        cv2.imwrite(
            overlay_path,
            overlay
        )


        # ----------------------------------------------------
        # Tumor detection
        # ----------------------------------------------------

        tumor_detected = bool(
            tumor_area > 0
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,
            
            "tumor_type": tumor_type,

            "tumor_detected":
                tumor_detected,

            "tumor_area_percent":
                round(
                    float(tumor_area),
                    2
                ),

            "mask_url":
                "/result/mask/"
                + mask_filename,

            "overlay_url":
                "/result/overlay/"
                + overlay_filename
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# GET MASK
# ============================================================

@prediction_bp.route(
    "/result/mask/<filename>",
    methods=["GET"]
)
def get_mask(filename):

    path = os.path.join(
        MASK_OUTPUT_DIR,
        filename
    )

    if not os.path.exists(path):

        return jsonify({
            "error": "Mask not found"
        }), 404


    return send_file(
        path,
        mimetype="image/png"
    )


# ============================================================
# GET OVERLAY
# ============================================================

@prediction_bp.route(
    "/result/overlay/<filename>",
    methods=["GET"]
)
def get_overlay(filename):

    path = os.path.join(
        OVERLAY_OUTPUT_DIR,
        filename
    )

    if not os.path.exists(path):

        return jsonify({
            "error": "Overlay not found"
        }), 404


    return send_file(
        path,
        mimetype="image/png"
    )
