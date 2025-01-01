from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
from sqlalchemy import DECIMAL
import pymysql
from decimal import Decimal
import streamlit as st
import os
import pathlib
import textwrap
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql://root:@localhost/KSA_Stock_Project'
db = SQLAlchemy(app)
genai.configure(api_key="AIzaSyA8CuC1z-d7kSK00dcve5ZorUnGFRoDuI0")

# Database Models
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
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)

class Stocks(db.Model):
    stock_id = db.Column(db.Integer, primary_key=True)
    ticker_symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(100), nullable=False)
    current_price = db.Column(DECIMAL(10,2), nullable=False)
    sector = db.Column(db.String(50))
    listing_date = db.Column(db.Date)
    high_price_52w = db.Column(DECIMAL(10,2))
    low_price_52w = db.Column(DECIMAL(10,2))
    updated_at = db.Column(db.DateTime)
    portfolios = db.relationship('Portfolio', backref='stock', lazy=True)

class StockPriceHistory(db.Model):
    __tablename__ = 'stock price history'
    price_history_id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    price = db.Column(db.DECIMAL(10, 2), nullable=False)
    date = db.Column(db.TIMESTAMP, nullable=False, server_default=db.func.current_timestamp())

class Portfolio(db.Model):
    portfolio_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_info.id'), nullable=False)
    # stock_id = db.Column(db.Integer, db.ForeignKey('Stocks.stock_id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False) 
    quantity = db.Column(db.Integer, nullable=False)
    purchase_price = db.Column(DECIMAL(10,2), nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)


def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(question)
    return response.text
 
# Routes
@app.route('/')
def index():
    return render_template('index.html')

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
        
        entry = User_info(
            name=first_name, Surname=surname, DOB_Day=day, DOB_Month=month,
            DOB_YEAR=year, Gender=gender, Mobile_Number_Or_Email=email, Password=password
        )
        db.session.add(entry)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_phone = request.form.get('email_phone')
        password = request.form.get('password')
        
        user = User_info.query.filter_by(Mobile_Number_Or_Email=email_or_phone, Password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['user_name'] = f"{user.name} {user.Surname}"
            flash('Login successful!', 'success')
            return redirect(url_for('portfolio'))
        else:
            flash('Invalid email/phone number or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/portfolio")
def portfolio():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Fetch user's portfolio with stock details
    portfolio_items = db.session.query(
        Portfolio, Stocks
    ).join(
        Stocks, Portfolio.stock_id == Stocks.stock_id
    ).filter(
        Portfolio.user_id == session['user_id']
    ).all()
    stock_prices = db.session.query(
        StockPriceHistory.stock_id, StockPriceHistory.price
    ).filter(
        StockPriceHistory.date == '2024-12-20'
    ).all()
    price_dict = {stock_id: price for stock_id, price in stock_prices}
    # Calculate total portfolio value and returns
    total_value = Decimal('0')
    
    # Format portfolio data for template
    holdings = []
    total_value=0
    for portfolio, stocks in portfolio_items:
        current_price = Decimal(price_dict.get(stocks.stock_id, 100))  # Fallback if price not found
        holding_value = current_price * Decimal(portfolio.quantity)
        total_value += holding_value
    for portfolio, stocks in portfolio_items:
        current_price = Decimal(price_dict.get(stocks.stock_id, 100))  # Fallback if price not found
        current_value = current_price * Decimal(portfolio.quantity)
        purchase_value = Decimal(portfolio.purchase_price) * Decimal(portfolio.quantity)
        return_pct = ((current_value - purchase_value) / purchase_value * 100) if purchase_value > 0 else 0
        weight = (current_value / total_value * 100) if total_value > 0 else 0
        
        holdings.append({
            'stock_name': stocks.company_name,
            'shares': portfolio.quantity,
            'avg_price': float(portfolio.purchase_price),
            'current_price': float(current_price),
            'weight': float(weight),
            'return': float(return_pct)
        })

    return render_template(
        "portfolio.html",
        user_name=session.get('user_name'),
        total_value=float(total_value),
        holdings=holdings
    )
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return {"response": "Invalid input. Please provide a valid query."}, 400

        query = data.get('query')
        if not query:
            return jsonify({"response": "Please provide a valid question."})

        try:
            response = get_gemini_response(query)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"response": f"Error: {str(e)}"})
    except Exception as e:
        print(f"Server Error: {str(e)}")  # For debugging
        return jsonify({"response": "Internal server error"}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)