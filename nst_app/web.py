from pathlib import Path
from uuid import uuid4

from flask import Blueprint, current_app, render_template, request, send_from_directory
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from nst_app.forms import CompareForm, UploadForm


bp = Blueprint("web", __name__)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _render_page(**context):
    defaults = dict(
        form=UploadForm(), compare_form=CompareForm(prefix="compare-"),
        result_image=None, content_image=None, style_image=None, error=None,
        comparison=None, comparison_content=None, compare_error=None,
        comparison_max_size=current_app.config["COMPARISON_IMAGE_SIZE"],
    )
    defaults.update(context)
    return render_template("index.html", **defaults)


def _save_upload(upload, role):
    original = Path(secure_filename(upload.filename))
    extension = original.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Images must be PNG or JPEG files.")

    stem = original.stem or role
    filename = f"{role}_{stem}_{uuid4().hex[:8]}{extension}"
    upload.save(Path(current_app.config["UPLOAD_FOLDER"]) / filename)
    return filename


@bp.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            if form.content.data and form.content.data.filename:
                content_filename = _save_upload(form.content.data, "content")
                form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            if form.style.data and form.style.data.filename:
                style_filename = _save_upload(form.style.data, "style")
                form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            if not content_filename or not style_filename:
                raise ValueError("Please upload both content and style images.")

            upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
            content_image = Image.open(upload_folder / content_filename).convert("RGB")
            style_image = Image.open(upload_folder / style_filename).convert("RGB")
            service = current_app.extensions["style_transfer"]
            stylized_image = service.stylize(
                content_image,
                style_image,
                float(form.alpha.data),
            )

            result_image = f"stylized_{Path(content_filename).stem}_{uuid4().hex[:8]}.jpg"
            service.save_image(stylized_image, upload_folder / result_image)
        except Exception as exc:
            error = str(exc)

    return _render_page(
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@bp.route("/compare", methods=["GET", "POST"])
def compare():
    form = CompareForm(prefix="compare-")
    result = None
    content_filename = None
    error = None
    if form.validate_on_submit():
        try:
            with Image.open(form.content.data.stream) as uploaded:
                content = ImageOps.exif_transpose(uploaded).convert("RGB")
            form.content.data.stream.seek(0)
            content_filename = _save_upload(form.content.data, "comparison")
            upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
            service = current_app.extensions["comparison"]
            outputs = service.compare(content)
            result = {"size": outputs["size"]}
            for method in ("adain", "fast"):
                filename = f"compare_{method}_{uuid4().hex}.png"
                service.adain.save_image(outputs[method]["image"], upload_folder / filename)
                result[method] = {"filename": filename, "seconds": outputs[method]["seconds"]}
        except Exception:
            current_app.logger.exception("Comparison failed")
            error = "We couldn't compare this image. Please try a valid PNG or JPEG, or check that the comparison model is installed."
            result = None
    elif request.method == "POST":
        error = " ".join(message for messages in form.errors.values() for message in messages)
    return _render_page(compare_form=form, comparison=result,
                        comparison_content=content_filename, compare_error=error)


@bp.route("/comparison-style")
def comparison_style():
    style = Path(current_app.config["COMPARISON_STYLE_PATH"])
    return send_from_directory(style.parent, style.name)


@bp.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@bp.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(current_app.config["EXAMPLES_FOLDER"], filename)
