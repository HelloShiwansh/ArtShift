import os
from pathlib import Path

from flask import Flask


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app(test_config=None):
    app = Flask(
        __name__,
        instance_path=str(PROJECT_ROOT / "instance"),
        instance_relative_config=True,
    )
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key"),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        UPLOAD_FOLDER=str(PROJECT_ROOT / "instance" / "uploads"),
        EXAMPLES_FOLDER=str(PROJECT_ROOT / "assets" / "examples"),
        VGG_MODEL_PATH=str(PROJECT_ROOT / "models" / "vgg_normalised.pth"),
        DECODER_MODEL_PATH=str(PROJECT_ROOT / "models" / "decoder_final.pth"),
    )

    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    from nst_app.inference import StyleTransferService
    from nst_app.web import bp

    app.extensions["style_transfer"] = StyleTransferService(
        app.config["VGG_MODEL_PATH"],
        app.config["DECODER_MODEL_PATH"],
    )
    app.register_blueprint(bp)

    return app
