from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://username:password@localhost/db_name'
# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for signup page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Route for login page
@app.route('/login')
def login():
    return render_template('..\templates\login.html')

if __name__ == '__main__':
    app.run(debug=True)
