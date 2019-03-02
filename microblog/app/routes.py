from flask import render_template,flash,redirect,url_for,jsonify
from app import app
from app.urlform import UrlForm
from app.forms import LoginForm
from app.bigboi import *

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html',title = 'Home')

@app.route('/attendance&marks', methods=['POST','GET'])
def info():
	url1= request.json['url']
	resp = login({"url":url})
	return jsonify(resp)

@app.route('/result', methods=['GET','POST'])
def result():
	return render_template('result.html',title='Result')

@app.route('/urlfor', methods=['GET','POST'])
def logine():
	form = UrlForm()
	if form.validate_on_submit():
		flash('The URL entered is {}'.format(form.url.data))
		return redirect(url_for('result'))
	return render_template('urlf.html', title = 'Enter The Drive Link', form = form)

@app.route('/login', methods=['GET','POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		name = process(form.username.data)
		flash('Login requested for user {}, remember_me={}'.format(name, form.remember_me.data))
		return redirect('urlfor')
	return render_template('login.html', title = 'Sign In', form = form)