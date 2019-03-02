from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class UrlForm(FlaskForm):
	url = StringField('Url', validators = [DataRequired()])
	submit = SubmitField('Submit')
