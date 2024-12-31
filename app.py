from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pymysql

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:@localhost/KSA_Stock_Project'
db = SQLAlchemy(app)

# Database model for user information
class User_info(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    Surname = db.Column(db.String(120), nullable=False)
    DOB_Day = db.Column(db.Integer, nullable=False)
    DOB_Month = db.Column(db.String(120), nullable=False)
    DOB_YEAR = db.Column(db.Integer, nullable=False)
    Gender = db.Column(db.String(120), nullable=False)
    Mobile_Number_Or_Email = db.Column(db.String(120), nullable=False)
    Password = db.Column(db.String(120), nullable=False)

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        surname = request.form.get('surname')
        day = int(request.form.get('day'))
        year = int(request.form.get('year'))
        month = request.form.get('month')
        gender = request.form.get('gender')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Add user to the database
        entry = User_info(
            name=first_name, Surname=surname, DOB_Day=day, DOB_Month=month,
            DOB_YEAR=year, Gender=gender, Mobile_Number_Or_Email=email, Password=password
        )
        db.session.add(entry)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_phone = request.form.get('email_phone')
        password = request.form.get('password')
        
        # Check if the user exists in the database
        user = User_info.query.filter_by(Mobile_Number_Or_Email=email_or_phone, Password=password).first()
        
        if user:
            flash('Login successful!', 'success')
            return redirect(url_for('portfolio'))  # Redirect to the home page or dashboard
        else:
            flash('Invalid email/phone number or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

if __name__ == '__main__':
    # Use application context to create database tables
    with app.app_context():
        db.create_all()
    app.run(debug=True)
