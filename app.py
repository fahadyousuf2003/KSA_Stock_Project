from flask import Flask, render_template, request, redirect, url_for, flash, session
from sqlalchemy import func
from flask_sqlalchemy import SQLAlchemy
import spacy
import re
from flask import jsonify
from sqlalchemy import DECIMAL
import pymysql
from decimal import Decimal
import streamlit as st
import os
import io
from contextlib import redirect_stdout
import json
import pathlib
import textwrap
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from datetime import date
import streamlit as st
import traceback
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from IPython.display import display
from IPython.display import Markdown
import pinecone
nlp = spacy.load("en_core_web_sm")
pymysql.install_as_MySQLdb()

pc = Pinecone(api_key="pcsk_7YQanE_AP9y5db9N5vQaoUYC2h6bxvr92sEPyXzVBUcotcvBubtsEqsfkDaLQ6LrGwWRnw")
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
    current_price = db.Column(DECIMAL(10, 2), nullable=False)
    sector = db.Column(db.String(50))
    listing_date = db.Column(db.Date)
    high_price_52w = db.Column(DECIMAL(10, 2))
    low_price_52w = db.Column(DECIMAL(10, 2))
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
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    purchase_price = db.Column(DECIMAL(10, 2), nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)


class Funds(db.Model):
    __tablename__ = 'funds'

    fund_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_info.id'), nullable=False)
    available_balance = db.Column(db.Float, nullable=False, default=0.0)
    total_balance = db.Column(db.Float, nullable=False, default=0.0)
    last_updated = db.Column(db.DateTime, nullable=False)


# index_name = "combineddataindex"
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
#     namespace=""
# )

# template = """You are a stock information assistant that helps users find details about stocks and their portfolios.
#             Use the following pieces of context to answer the question at the end.
#             If any information is missing from the database, provide a general answer based on your knowledge.

#             {context}

#             Question: {question}
#             Your response:"""

# PROMPT = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

# llm = ChatGroq(
#     model_name="llama3-8b-8192",  # Change to an appropriate model for stock-related queries
#     api_key="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",
#     temperature=0
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )


def initialize_recommendation_system():
    try:
        # Initialize Groq
        groq_client = Groq(api_key="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")

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Pinecone
        pc = Pinecone(api_key="pcsk_7YQanE_AP9y5db9N5vQaoUYC2h6bxvr92sEPyXzVBUcotcvBubtsEqsfkDaLQ6LrGwWRnw")

        # Get the index for stock data
        index_name = "combineddataindex"  # Change to the stock-related index name
        index = pc.Index(index_name)

        # Check index stats
        index_stats = index.describe_index_stats()

        # Initialize vector store
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace=""
        )

        # Initialize LLM (for stock-related information)
        llm = ChatGroq(
            model_name="llama3-8b-8192",  # Change to an appropriate model for stock-related queries
            api_key="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",
            temperature=0
        )

        # Define prompt template for stock-related information
        template = """You are a stock information assistant that helps users find details about stocks and their portfolios.
        Use the following pieces of context to answer the question at the end.
        If any information is missing from the database, provide a general answer based on your knowledge.

        {context}

        Question: {question}
        Your response:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create QA chain for stock-related queries
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain
    except Exception as e:
        # Handle any exceptions that occur during the process
        return f"An error occurred while retrieving the context: {str(e)}"

# Function to retrieve ranked context based on user_id and query
def retrieve_ranked_context(user_id, query):
    try:
        # Retrieve top 100 matches from the vector store
        results = docsearch.similarity_search(query, k=100)

        # Filter results to only include those related to the given user_id
        user_specific_results = [doc for doc in results if doc.metadata.get('user_id') == user_id]

        # Sort by relevant fields (e.g., stock price or portfolio details)
        ranked_results = sorted(
            user_specific_results,
            key=lambda x: float(x.metadata.get("price", 0)),  # Sort by stock price or other relevant field
            reverse=True
        )

        # If no results are found for the user, provide a general answer
        if not ranked_results:
            return "Sorry, I couldn't find any specific information for this user. Here's some general information about stocks and portfolios."

        # Create context from the top 3 results
        context = "\n".join([doc.page_content for doc in ranked_results[:3]])  # Top 3 results
        return context

    except Exception as e:
        # Handle any exceptions that occur during the process
        return f"An error occurred while retrieving the context: {str(e)}"


def get_gemini_response(query, user_id=None):
    try:
        with io.StringIO() as buf, redirect_stdout(buf):
            # Retrieve context specific to the user
            context = retrieve_ranked_context(user_id, query)
            qa_input = {"context": context, "query": query}
            result = qa_chain(qa_input)

        # Return the final result without any intermediate prints
        return result['result']
    except Exception as e:
        # Handle any exceptions that occur during the process
        return {"error": f"An error occurred: {str(e)}"}


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
    total_value = 0
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
        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if user_funds:
            user_funds.total_balance = float(user_funds.available_balance) + float(total_value)
            db.session.commit()

    return render_template(
        "portfolio.html",
        user_name=session.get('user_name'),
        total_value=float(total_value),
        available_balance=user_funds.available_balance,
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
        user_id =  session['user_id']
        if not query:
            return jsonify({"response": "Please provide a valid question."})

        if 'buy' in query.lower():
            stock_name, quantity = extract_stock_info(query)
            if stock_name and quantity:
                # Call the buy_stock function with extracted stock name and quantity
                return buy_stock_logic(stock_name, quantity)
            else:
                return jsonify({"response": "Unable to extract stock name and quantity from the query."})
        if 'sell' in query.lower():
            stock_name, quantity = extract_stock_info(query)
            if stock_name and quantity:
                return sell_stock_logic(stock_name, quantity)
            else:
                return jsonify({"response": "Unable to extract stock name and quantity from the query."})

        try:
            response = get_gemini_response(query, user_id)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"response": f"Error: {str(e)}"})
    except Exception as e:
        print(f"Server Error: {str(e)}")  # For debugging
        return jsonify({"response": "Internal server error"}), 500
     
@app.route('/tradingview')
def tradingview():
    return render_template('tradingview.html')

@app.route('/buy_stock', methods=['GET', 'POST'])
def buy_stock():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        stock_name = request.form.get('stock_name')
        quantity = int(request.form.get('quantity'))
        stock = Stocks.query.filter_by(company_name=stock_name).first()
        if not stock:
            flash('Stock not found.', 'danger')
            return redirect(url_for('portfolio'))
        purchase_price = stock.current_price

        # Fetch stock details by name

        # Calculate total cost
        total_cost = quantity * purchase_price

        # Fetch user's funds
        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if not user_funds or user_funds.available_balance < float(total_cost):
            flash('Insufficient funds.', 'danger')
            return redirect(url_for('portfolio'))

        # Update portfolio
        portfolio_entry = Portfolio.query.filter_by(user_id=session['user_id'], stock_id=stock.stock_id).first()
        if portfolio_entry:           
            portfolio_entry.purchase_price = (
                (portfolio_entry.purchase_price * portfolio_entry.quantity) + total_cost
            ) / (portfolio_entry.quantity + quantity)
            portfolio_entry.quantity += quantity
        else:
            portfolio_entry = Portfolio(
                user_id=session['user_id'],
                stock_id=stock.stock_id,
                quantity=quantity,
                purchase_price=purchase_price,
                purchase_date=date.today()
            )
            db.session.add(portfolio_entry)

        # Deduct funds
        user_funds.available_balance -= float(total_cost)
        try:
            db.session.commit()
            flash('Stock purchased successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')

        return redirect(url_for('portfolio'))

    return render_template('buy_stock.html')
def extract_stock_info(query):
    """Extract stock name and quantity from the query using NLP."""
    stock_list = [
        "Saudi Aramco", "SARCO", "PETRO RABIGH", "ARABIAN DRILLING", 
        "ADES", "BAHRI", "ALDREES"
    ]
    query_lower = query.lower()
    for stock in stock_list:
        if stock.lower() in query_lower:
            stock_name = stock
            break 
    regex = r'(\d+)\s*(share|shares)\s*of\s*([A-Za-z\s]+)'
    match = re.search(regex, query_lower)
    if match:
        quantity = int(match.group(1))
        # stock_name = match.group(3).strip().title()  # Capitalize stock name correctly
        return stock_name, quantity
    else:
        # Fallback to spaCy for stock name extraction if regex fails
        doc = nlp(query)
        stock_name = None
        quantity = None

    # Extract entities related to stocks (assuming stock names are proper nouns)
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Assuming stock names are recognized as organizations
            stock_name = ent.text
        elif ent.label_ == "CARDINAL":  # Assuming quantity is a number
            quantity = int(ent.text)

    # Return extracted stock name and quantity
    return stock_name, quantity
def buy_stock_logic(stock_name, quantity):
    """Handle the stock purchase logic."""
    # This is a simplified version of your buy_stock function, modify as per your logic
    stock = Stocks.query.filter_by(company_name=stock_name).first()
    if not stock:
        return jsonify({"response": "Stock not found."}), 404

    purchase_price = stock.current_price
    total_cost = quantity * purchase_price

    user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
    if not user_funds or user_funds.available_balance < float(total_cost):
        return jsonify({"response": "Insufficient funds."}), 400

    portfolio_entry = Portfolio.query.filter_by(user_id=session['user_id'], stock_id=stock.stock_id).first()
    if portfolio_entry:
        portfolio_entry.purchase_price = (
            (portfolio_entry.purchase_price * portfolio_entry.quantity) + total_cost
        ) / (portfolio_entry.quantity + quantity)
        portfolio_entry.quantity += quantity
    else:
        portfolio_entry = Portfolio(
            user_id=session['user_id'],
            stock_id=stock.stock_id,
            quantity=quantity,
            purchase_price=purchase_price,
            purchase_date=date.today()
        )
        db.session.add(portfolio_entry)

    user_funds.available_balance -= float(total_cost)
    try:
        db.session.commit()
        return jsonify({"response": "Stock purchased successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"response": f"Error: {str(e)}"}), 500
    

@app.route('/sell_stock', methods=['GET', 'POST'])
def sell_stock():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        stock_name = request.form.get('stock_name')
        quantity = int(request.form.get('quantity'))

        # Fetch stock details by name
        stock = Stocks.query.filter(func.lower(Stocks.company_name) == stock_name.lower()).first()
        if not stock:
            flash('Stock not found.', 'danger')
            return redirect(url_for('portfolio'))

        # Check user's portfolio
        portfolio_entry = Portfolio.query.filter_by(user_id=session['user_id'], stock_id=stock.stock_id).first()
        if not portfolio_entry or portfolio_entry.quantity < quantity:
            flash('Insufficient stock quantity.', 'danger')
            return redirect(url_for('portfolio'))

        # Calculate the total sale amount
        sale_price = stock.current_price
        total_sale = quantity * sale_price

        # Update portfolio
        if portfolio_entry.quantity == quantity:
            db.session.delete(portfolio_entry)  # Remove entry if all stocks are sold
        else:
            portfolio_entry.quantity -= quantity

        # Update user's funds
        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if user_funds:
            user_funds.available_balance += float(total_sale)
        else:
            flash('Error updating funds.', 'danger')
            return redirect(url_for('portfolio'))

        # Commit changes
        try:
            db.session.commit()
            flash('Stock sold successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')

        return redirect(url_for('portfolio'))

    return render_template('sell_stock.html')

def sell_stock_logic(stock_name, quantity):
    """Handle the stock sale logic."""
    stock = Stocks.query.filter_by(company_name=stock_name).first()
    if not stock:
        return jsonify({"response": "Stock not found."}), 404

    portfolio_entry = Portfolio.query.filter_by(user_id=session['user_id'], stock_id=stock.stock_id).first()
    if not portfolio_entry or portfolio_entry.quantity < quantity:
        return jsonify({"response": "Insufficient stock quantity."}), 400

    sale_price = stock.current_price
    total_sale = quantity * sale_price

    if portfolio_entry.quantity == quantity:
        db.session.delete(portfolio_entry)
    else:
        portfolio_entry.quantity -= quantity

    user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
    if user_funds:
        user_funds.available_balance += float(total_sale)
    else:
        return jsonify({"response": "Error updating funds."}), 500

    try:
        db.session.commit()
        return jsonify({"response": "Stock sold successfully!"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)