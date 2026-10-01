from flask import Flask
from flask_cors import CORS

from routes.prediction import (
    prediction_bp
)

from routes.health import (
    health_bp
)


app = Flask(
    __name__
)

CORS(app)


app.register_blueprint(
    prediction_bp
)

app.register_blueprint(
    health_bp
)


@app.route("/")
def home():

    return {
        "message":
            "Brain MRI Segmentation API",
        "status":
            "running"
    }
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )