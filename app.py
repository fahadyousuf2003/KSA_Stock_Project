from flask import Flask, render_template,request
from flask_sqlalchemy import SQLAlchemy
import pymysql

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:@localhost/KSA_Stock_Project'
db = SQLAlchemy(app)
#id,name,Surname,DOB_Day,DOB_Month,DOB_YEAR,Gender,Mobile_Number_Or_Email,Password

class User_info(db.Model):
    id= db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(120),nullable=False)
    Surname=db.Column(db.String(120),nullable=False)
    DOB_Day=db.Column(db.Integer,nullable=False)
    DOB_Month=db.Column(db.String(120),nullable=False)
    DOB_YEAR=db.Column(db.Integer,nullable=False)
    Gender=db.Column(db.String(120),nullable=False)
    Mobile_Number_Or_Email=db.Column(db.String(120),nullable=False)
    Password=db.Column(db.String(120),nullable=False)
# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for signup page
@app.route('/signup',methods=['GET','POST'])
def signup():
    if(request.method=='POST'):
        first_name=request.form.get('first_name')
        surname=request.form.get('surname')
        day = int(request.form.get('day'))
        year = int(request.form.get('year'))
        month=request.form.get('month')
        gender=request.form.get('gender')
        email=request.form.get('email')
        password=request.form.get('password')
        # id,name,Surname,DOB_Day,DOB_Month,DOB_YEAR,Gender,Mobile_Number_Or_Email,Password
        entry=User_info(name=first_name,Surname=surname,DOB_Day=day,DOB_Month=month,DOB_YEAR=year,Gender=gender,Mobile_Number_Or_Email=email,Password=password)
        db.session.add(entry)
        db.session.commit()
    return render_template('signup.html')

# Route for login page
@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    # Use application context to create database tables
    with app.app_context():
        db.create_all()
    app.run(debug=True)