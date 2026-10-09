from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import FileField, FloatField, HiddenField, SubmitField


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


class CompareForm(FlaskForm):
    content = FileField("Content image", validators=[
        FileRequired("Please choose a content image."),
        FileAllowed(["png", "jpg", "jpeg"], "Images must be PNG or JPEG files."),
    ])
    submit = SubmitField("Compare methods")
