from flask_wtf import FlaskForm
from wtforms import FileField, FloatField, HiddenField, SubmitField


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")
