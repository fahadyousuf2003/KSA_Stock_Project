from flask import Flask, render_template, request, redirect, url_for, flash, session
from sqlalchemy import func
from flask_sqlalchemy import SQLAlchemy
import spacy
import re
from flask import jsonify
from sqlalchemy import DECIMAL
from decimal import Decimal
import os
import io
from contextlib import redirect_stdout
from datetime import date
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
import yfinance as yf
import threading
from threading import Thread 
import time
from threading import Lock
import logging
from langchain_core.documents import Document
import pandas as pd
from decimal import Decimal, InvalidOperation
from datetime import datetime

latest_data = {}
historical_data = {}
data_lock = Lock()
nlp = spacy.load("en_core_web_sm")

pc = Pinecone(api_key="pcsk_4bw8VS_9RhtdAiBFfpZKN8p54VpYBVZVwBCGF7DCPKYVHVwVh6LWcfUCowPa6a1UEUwUCo")
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["SQLALCHEMY_DATABASE_URI"] = 'postgresql://fahad:Ux7CHDifnMzJMUDFQLBNZZkI1tOJ64NC@dpg-cu1ockt2ng1s73edt7r0-a.oregon-postgres.render.com/ksastockproject'
db = SQLAlchemy(app)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Database Models
class User_info(db.Model):
    __tablename__ = 'user_info'
    id = db.Column(db.Integer, primary_key=True, server_default=db.text("nextval('user_info_id_seq')"))
    name = db.Column(db.String(40), nullable=False)
    surname = db.Column(db.String(40), nullable=False)
    dob_day = db.Column(db.Integer, nullable=False)
    dob_month = db.Column(db.String(40), nullable=False)
    dob_year = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Text, nullable=False)
    mobile_number_or_email = db.Column(db.Text, nullable=False)
    password = db.Column(db.Text, nullable=False)
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)
    funds = db.relationship('Funds', backref='user', lazy=True)

class Stocks(db.Model):
    __tablename__ = 'stocks'
    stock_id = db.Column(db.Integer, primary_key=True, server_default=db.text("nextval('stocks_stock_id_seq')"))
    ticker_symbol = db.Column(db.String(120), nullable=False)
    company_name = db.Column(db.String(120), nullable=False)
    current_price = db.Column(db.DECIMAL(40, 2), nullable=False)
    sector = db.Column(db.String(120), nullable=False)
    listing_date = db.Column(db.TIMESTAMP, nullable=False, server_default=db.func.current_timestamp())
    high_price_52w = db.Column(db.DECIMAL(40, 2), nullable=False)
    low_price_52w = db.Column(db.DECIMAL(40, 2), nullable=False)
    updated_at = db.Column(db.TIMESTAMP, nullable=False, server_default=db.func.current_timestamp())
    portfolios = db.relationship('Portfolio', backref='stock', lazy=True)
    price_history = db.relationship('StockPriceHistory', backref='stock', lazy=True)

class StockPriceHistory(db.Model):
    __tablename__ = 'stock price history'  # Note: keeping the space as per SQL file
    price_history_id = db.Column(db.Integer, primary_key=True, server_default=db.text("nextval('stock_price_history_price_history_id_seq')"))
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    price = db.Column(db.DECIMAL(10, 2), nullable=False)
    date = db.Column(db.TIMESTAMP, nullable=False, server_default=db.func.current_timestamp())

class Portfolio(db.Model):
    __tablename__ = 'portfolio'
    portfolio_id = db.Column(db.Integer, primary_key=True, server_default=db.text("nextval('portfolio_portfolio_id_seq')"))
    user_id = db.Column(db.Integer, db.ForeignKey('user_info.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    purchase_price = db.Column(db.DECIMAL(10, 0), nullable=False)  # Changed to DECIMAL(10,0) as per SQL
    purchase_date = db.Column(db.TIMESTAMP, nullable=False, server_default=db.func.current_timestamp())

class Funds(db.Model):
    __tablename__ = 'funds'
    fund_id = db.Column(db.Integer, primary_key=True, server_default=db.text("nextval('funds_fund_id_seq')"))
    user_id = db.Column(db.Integer, db.ForeignKey('user_info.id'), nullable=False)
    available_balance = db.Column(db.DECIMAL(10, 2), nullable=False)
    total_balance = db.Column(db.DECIMAL(10, 2), nullable=False)
    last_updated = db.Column(db.DECIMAL(10, 2), nullable=False)  # Added missing column
    
combined_data2 = pd.read_pickle('combined_data2.pkl')
combined_data= pd.read_pickle('combined_data.pkl')
documents2 = [
    Document(
        page_content=row['combined_info'],
        metadata={
            'user_id': row['user_id'],
        }
    ) for _, row in combined_data2.iterrows()
]
documents = [
    Document(
        page_content=row['combined_info'],
        metadata={
            'user_id': row['user_id'],
        }
    ) for _, row in combined_data.iterrows()
]
# Create Document objects


# Example: Print a document
print(documents[0].page_content)
print(documents[0].metadata)
# Set the API key as an environment variable
os.environ["PINECONE_API_KEY"] = "pcsk_7YQanE_AP9y5db9N5vQaoUYC2h6bxvr92sEPyXzVBUcotcvBubtsEqsfkDaLQ6LrGwWRnw"

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define LLM using Groq (Llama model)
llm = ChatGroq(
    model_name="llama3-8b-8192",  # or another Llama model available
    api_key="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",
    temperature=0
)
# Define custom prompt template for stock-related information specific to the user
template = """You are a stock information assistant that helps users find details about stocks and their portfolios.
Use the following pieces of context to answer the question at the end.
If any information is missing from the database, provide a general answer based on your knowledge.
answer the question 3-4 lines only

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            # Create new user
            user = User_info(
                name=request.form.get('first_name'),
                surname=request.form.get('surname'),
                dob_day=int(request.form.get('day')),
                dob_month=request.form.get('month'),
                dob_year=int(request.form.get('year')),
                gender=request.form.get('gender'),
                mobile_number_or_email=request.form.get('email'),
                password=request.form.get('password')  # In production, hash this password!
            )
            
            # Create initial funds entry for new user
            funds = Funds(
                user_id=user.id,
                available_balance=Decimal('0'),
                total_balance=Decimal('0'),
                last_updated=Decimal(datetime.now().year)  # Current year as decimal
            )
            
            db.session.add(user)
            db.session.add(funds)
            db.session.commit()
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating account: {str(e)}', 'danger')
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            user = User_info.query.filter_by(
                mobile_number_or_email=request.form.get('email_phone'),
                password=request.form.get('password')  # In production, verify hashed password!
            ).first()

            if user:
                session['user_id'] = user.id
                session['user_name'] = f"{user.name} {user.surname}"
                flash('Login successful!', 'success')
                return redirect(url_for('portfolio'))
            
            flash('Invalid email/phone number or password.', 'danger')
            
        except Exception as e:
            flash(f'Login error: {str(e)}', 'danger')
            
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route("/portfolio")
def portfolio():
    if 'user_id' not in session:
        flash('Please login to view your portfolio.', 'warning')
        return redirect(url_for('login'))
    
    try:
        # Get portfolio items with stock information
        portfolio_items = (
            db.session.query(Portfolio, Stocks)
            .join(Stocks, Portfolio.stock_id == Stocks.stock_id)
            .filter(Portfolio.user_id == session['user_id'])
            .all()
        )
        
        # Fetch current prices from Yahoo Finance
        stock_symbols = [stocks.ticker_symbol for _, stocks in portfolio_items]
        current_prices = {}
        
        if stock_symbols:
            try:
                stock_data = yf.download(stock_symbols, period="1d", interval="1d", progress=False)
                if not stock_data.empty:
                    if len(stock_symbols) == 1:
                        current_prices[stock_symbols[0]] = float(stock_data['Close'].iloc[-1])
                    else:
                        current_prices = stock_data['Close'].iloc[-1].to_dict()
            except Exception as e:
                app.logger.error(f"YFinance error: {str(e)}")
        
        # Calculate portfolio statistics
        total_value = Decimal('0')
        holdings = []
        
        for portfolio, stock in portfolio_items:
            try:
                # Get current price from YFinance or fallback to 52-week high
                current_price = Decimal(str(current_prices.get(stock.ticker_symbol, 0)))
                if current_price == 0:
                    current_price = stock.high_price_52w or Decimal('0')
                
                quantity = Decimal(str(portfolio.quantity))
                purchase_price = Decimal(str(portfolio.purchase_price))
                current_value = current_price * quantity
                purchase_value = purchase_price * quantity
                
                # Calculate return percentage
                return_pct = ((current_value - purchase_value) / purchase_value * 100) if purchase_value > 0 else Decimal('0')
                
                total_value += current_value
                
                holdings.append({
                    'stock_name': stock.company_name,
                    'ticker': stock.ticker_symbol,
                    'shares': float(quantity),
                    'avg_price': float(purchase_price),
                    'current_price': float(current_price),
                    'current_value': float(current_value),
                    'return': float(return_pct),
                    'weight': 0  # Will be calculated after total is known
                })
                
            except (TypeError, ValueError, InvalidOperation) as e:
                app.logger.error(f"Calculation error for {stock.ticker_symbol}: {str(e)}")
        
        # Calculate weights after total value is known
        for holding in holdings:
            holding['weight'] = float((Decimal(str(holding['current_value'])) / total_value * 100) if total_value > 0 else 0)
        
        # Update user funds
        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if user_funds:
            try:
                user_funds.total_balance = float(Decimal(str(user_funds.available_balance)) + total_value)
                user_funds.last_updated = Decimal(datetime.now().year)
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Funds update error: {str(e)}")
                db.session.rollback()
        
        return render_template(
            "portfolio.html",
            user_name=session.get('user_name'),
            total_value=float(total_value),
            available_balance=float(user_funds.available_balance) if user_funds else 0,
            holdings=sorted(holdings, key=lambda x: x['current_value'], reverse=True)
        )
        
    except Exception as e:
        flash(f'Error loading portfolio: {str(e)}', 'danger')
        return redirect(url_for('index'))
    
os.environ["PINECONE_API_KEY"] = "pcsk_7YQanE_AP9y5db9N5vQaoUYC2h6bxvr92sEPyXzVBUcotcvBubtsEqsfkDaLQ6LrGwWRnw"

def retrieve_ranked_context(user_id, query):
    
# Perform the join
    if user_id == 2:
        index_name = "ksa-user-2"
        docsearch = PineconeVectorStore.from_documents(
            documents=documents2,
            embedding=embeddings,
            index_name=index_name
        )
    if user_id == 1:
        index_name = "ksa-user-1"
        docsearch = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
    # Filter the documents based on user_id
    results = docsearch.similarity_search(query, k=100)  # Retrieve top 100 matches
    
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
def chat(user_id,qa_input):
    if user_id == 2:
        index_name = "ksa-user-2"
    if user_id == 1:
        index_name = "ksa-user-1"
    
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa_chain(qa_input)
    return result
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"response": "Invalid input. Please provide a valid query."}), 400

        query = data.get('query')
        user_id = session.get('user_id', None)
        if not query:
            return jsonify({"response": "Please provide a valid question."})

        if 'buy' in query.lower():
            stock_name, quantity = extract_stock_info(query)
            if stock_name and quantity:
                return buy_stock_logic(stock_name, quantity)
            else:
                return jsonify({"response": "Unable to extract stock name and quantity from the query."})
        
        if 'sell' in query.lower():
            stock_name, quantity = extract_stock_info(query)
            if stock_name and quantity:
                return sell_stock_logic(stock_name, quantity)
            else:
                return jsonify({"response": "Unable to extract stock name and quantity from the query."})
        response = get_stock_info(user_id, query)
        print (response)
        if isinstance(response, dict):  # Handle dictionary response
            return jsonify(response)
        else:
            return jsonify({"response": response})
    except Exception as e:
        print(f"Server Error: {str(e)}")  # For debugging
        return jsonify({"response": "Internal server error"}), 500
     
@app.route('/tradingview')
def tradingview():
    return render_template('tradingview.html')
    
from sqlalchemy.orm import joinedload
import pandas as pd
def get_stock_info(user_id, query):
    with io.StringIO() as buf, redirect_stdout(buf):

        # Retrieve context specific to the user
        context = retrieve_ranked_context(user_id, query)
        qa_input = {"context": context, "query": query}
        result = chat(user_id, qa_input)

    # Return the final result without any intermediate prints
    return result['result']

@app.route('/buy_stock', methods=['GET', 'POST'])
def buy_stock():
    if 'user_id' not in session:
        flash('Please login to buy stocks.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            stock_name = request.form.get('stock_name')
            quantity = int(request.form.get('quantity'))
            
            # Fetch stock details
            stock = Stocks.query.filter(func.lower(Stocks.company_name) == stock_name.lower()).first()
            if not stock:
                flash('Stock not found.', 'danger')
                return redirect(url_for('portfolio'))
            
            # Convert to Decimal for precise calculations
            purchase_price = Decimal(str(stock.current_price))
            quantity_decimal = Decimal(str(quantity))
            total_cost = purchase_price * quantity_decimal

            # Check user funds
            user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
            if not user_funds or user_funds.available_balance < total_cost:
                flash('Insufficient funds.', 'danger')
                return redirect(url_for('portfolio'))

            # Update portfolio
            portfolio_entry = Portfolio.query.filter_by(
                user_id=session['user_id'], 
                stock_id=stock.stock_id
            ).first()

            if portfolio_entry:
                # Calculate new average purchase price
                old_total = portfolio_entry.purchase_price * Decimal(str(portfolio_entry.quantity))
                new_total = old_total + total_cost
                new_quantity = portfolio_entry.quantity + quantity
                portfolio_entry.purchase_price = new_total / Decimal(str(new_quantity))
                portfolio_entry.quantity = new_quantity
            else:
                portfolio_entry = Portfolio(
                    user_id=session['user_id'],
                    stock_id=stock.stock_id,
                    quantity=quantity,
                    purchase_price=purchase_price,
                    purchase_date=datetime.now()
                )
                db.session.add(portfolio_entry)

            # Update funds
            user_funds.available_balance -= total_cost
            user_funds.last_updated = Decimal(datetime.now().year)

            db.session.commit()
            flash('Stock purchased successfully!', 'success')

        except ValueError as e:
            db.session.rollback()
            flash('Invalid quantity specified.', 'danger')
        except Exception as e:
            db.session.rollback()
            flash(f'Error during purchase: {str(e)}', 'danger')

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
    try:
        stock = Stocks.query.filter(func.lower(Stocks.company_name) == stock_name.lower()).first()
        if not stock:
            return jsonify({"response": "Stock not found."}), 404

        # Convert to Decimal for precise calculations
        purchase_price = Decimal(str(stock.current_price))
        quantity_decimal = Decimal(str(quantity))
        total_cost = purchase_price * quantity_decimal

        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if not user_funds or user_funds.available_balance < total_cost:
            return jsonify({"response": "Insufficient funds."}), 400

        portfolio_entry = Portfolio.query.filter_by(
            user_id=session['user_id'], 
            stock_id=stock.stock_id
        ).first()

        if portfolio_entry:
            old_total = portfolio_entry.purchase_price * Decimal(str(portfolio_entry.quantity))
            new_total = old_total + total_cost
            new_quantity = portfolio_entry.quantity + quantity
            portfolio_entry.purchase_price = new_total / Decimal(str(new_quantity))
            portfolio_entry.quantity = new_quantity
        else:
            portfolio_entry = Portfolio(
                user_id=session['user_id'],
                stock_id=stock.stock_id,
                quantity=quantity,
                purchase_price=purchase_price,
                purchase_date=datetime.now()
            )
            db.session.add(portfolio_entry)

        user_funds.available_balance -= total_cost
        user_funds.last_updated = Decimal(datetime.now().year)

        db.session.commit()
        return jsonify({"response": "Stock purchased successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"response": f"Error: {str(e)}"}), 500
    
@app.route('/sell_stock', methods=['GET', 'POST'])
def sell_stock():
    if 'user_id' not in session:
        flash('Please login to sell stocks.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            stock_name = request.form.get('stock_name')
            quantity = int(request.form.get('quantity'))

            stock = Stocks.query.filter(func.lower(Stocks.company_name) == stock_name.lower()).first()
            if not stock:
                flash('Stock not found.', 'danger')
                return redirect(url_for('portfolio'))

            portfolio_entry = Portfolio.query.filter_by(
                user_id=session['user_id'], 
                stock_id=stock.stock_id
            ).first()

            if not portfolio_entry or portfolio_entry.quantity < quantity:
                flash('Insufficient stock quantity.', 'danger')
                return redirect(url_for('portfolio'))

            # Convert to Decimal for precise calculations
            sale_price = Decimal(str(stock.current_price))
            quantity_decimal = Decimal(str(quantity))
            total_sale = sale_price * quantity_decimal

            if portfolio_entry.quantity == quantity:
                db.session.delete(portfolio_entry)
            else:
                portfolio_entry.quantity -= quantity

            user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
            if user_funds:
                user_funds.available_balance += total_sale
                user_funds.last_updated = Decimal(datetime.now().year)
            else:
                flash('Error updating funds.', 'danger')
                return redirect(url_for('portfolio'))

            db.session.commit()
            flash('Stock sold successfully!', 'success')

        except ValueError as e:
            db.session.rollback()
            flash('Invalid quantity specified.', 'danger')
        except Exception as e:
            db.session.rollback()
            flash(f'Error during sale: {str(e)}', 'danger')

        return redirect(url_for('portfolio'))

    return render_template('sell_stock.html')

def sell_stock_logic(stock_name, quantity):
    """Handle the stock sale logic."""
    try:
        stock = Stocks.query.filter(func.lower(Stocks.company_name) == stock_name.lower()).first()
        if not stock:
            return jsonify({"response": "Stock not found."}), 404

        portfolio_entry = Portfolio.query.filter_by(
            user_id=session['user_id'], 
            stock_id=stock.stock_id
        ).first()

        if not portfolio_entry or portfolio_entry.quantity < quantity:
            return jsonify({"response": "Insufficient stock quantity."}), 400

        # Convert to Decimal for precise calculations
        sale_price = Decimal(str(stock.current_price))
        quantity_decimal = Decimal(str(quantity))
        total_sale = sale_price * quantity_decimal

        if portfolio_entry.quantity == quantity:
            db.session.delete(portfolio_entry)
        else:
            portfolio_entry.quantity -= quantity

        user_funds = Funds.query.filter_by(user_id=session['user_id']).first()
        if user_funds:
            user_funds.available_balance += total_sale
            user_funds.last_updated = Decimal(datetime.now().year)
        else:
            return jsonify({"response": "Error updating funds."}), 500

        db.session.commit()
        return jsonify({"response": "Stock sold successfully!"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"response": f"Error: {str(e)}"}), 500


def fetch_historical_data(symbol):
    """Fetch historical data for a specific stock symbol."""
    global historical_data
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo", interval="1h")
        if not hist.empty:
            with data_lock:
                historical_data[symbol] = [
                    {
                        "timestamp": row.name.isoformat(),
                        "open": round(row["Open"], 2),
                        "high": round(row["High"], 2),
                        "low": round(row["Low"], 2),
                        "close": round(row["Close"], 2),
                        "volume": int(row["Volume"])
                    }
                    for index, row in hist.iterrows()
                ]
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")

def fetch_realtime_data(symbol):
    """Fetch real-time data for a specific stock symbol."""
    global latest_data
    while True:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                last_row = hist.iloc[-1]
                with data_lock:
                    latest_data[symbol] = {
                        "timestamp": last_row.name.isoformat(),
                        "open": round(last_row["Open"], 2),
                        "high": round(last_row["High"], 2),
                        "low": round(last_row["Low"], 2),
                        "close": round(last_row["Close"], 2),
                        "volume": int(last_row["Volume"])
                    }
        except Exception as e:
            logging.error(f"Error fetching real-time data for {symbol}: {e}")
        time.sleep(1)

@app.route("/api/stock_data")
def get_stock_data():
    symbol = request.args.get('symbol', '2222.SR')  # Default to ARAMCO if no symbol provided
    
    with data_lock:
        return jsonify({
            "historical": historical_data.get(symbol, []),
            "latest": latest_data.get(symbol, {})
        })

def start_stock_threads():
    """Start threads for each stock symbol."""
    stock_symbols = [
        "2030.SR",  # SARCO
        "2222.SR",  # SAUDI ARAMCO
        "2380.SR",  # PETRO RABIGH
        "2381.SR",  # ARABIAN DRILLING
        "2382.SR",  # ADES
        "4030.SR",  # BAHRI
        "4200.SR"   # ALDREES
    ]
    
    for symbol in stock_symbols:
        # Start historical data thread
        threading.Thread(
            target=fetch_historical_data,
            args=(symbol,),
            daemon=True
        ).start()
        
        # Start real-time data thread
        threading.Thread(
            target=fetch_realtime_data,
            args=(symbol,),
            daemon=True
        ).start()
def monitor_stock_price(user_id, stock_id, target_price, action, quantity):
    """
    Monitor stock price and execute the action when the target price is reached.
    :param user_id: ID of the user
    :param stock_id: ID of the stock
    :param target_price: Target price for the action
    :param action: 'buy' or 'sell'
    :param quantity: Number of shares to buy/sell
    """
    while True:
        stock = Stocks.query.get(stock_id)
        if not stock:
            print(f"Stock with ID {stock_id} not found.")
            return
            
        current_price = stock.current_price
        
        if (action == 'buy' and current_price <= target_price) or (action == 'sell' and current_price >= target_price):
            # Execute the trade
            portfolio = Portfolio.query.filter_by(user_id=user_id, stock_id=stock_id).first()
            funds = Funds.query.filter_by(user_id=user_id).first()

            if action == 'buy':
                total_cost = current_price * quantity
                if funds.available_balance >= total_cost:
                    funds.available_balance -= total_cost
                    if portfolio:
                        # Update existing portfolio entry
                        new_total_quantity = portfolio.quantity + quantity
                        new_avg_price = ((portfolio.purchase_price * portfolio.quantity) + (current_price * quantity)) / new_total_quantity
                        portfolio.quantity = new_total_quantity
                        portfolio.purchase_price = new_avg_price
                    else:
                        # Create new portfolio entry
                        db.session.add(Portfolio(
                            user_id=user_id,
                            stock_id=stock_id,
                            quantity=quantity,
                            purchase_price=current_price,
                            purchase_date=date.today()
                        ))
                    flash(f"Bought {quantity} shares of {stock.company_name} at {current_price}", "success")
                else:
                    flash("Insufficient funds to buy stock.", "danger")

            elif action == 'sell':
                if portfolio and portfolio.quantity >= quantity:
                    total_sale = current_price * quantity
                    funds.available_balance += total_sale
                    portfolio.quantity -= quantity
                    if portfolio.quantity == 0:
                        db.session.delete(portfolio)
                    flash(f"Sold {quantity} shares of {stock.company_name} at {current_price}", "success")
                else:
                    flash("Insufficient stocks to sell.", "danger")

            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                flash(f"Error executing trade: {str(e)}", "danger")
            return

        time.sleep(10)  # Check every 10 seconds

@app.route('/stock_loss', methods=['GET', 'POST'])
def stop_loss():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        user_id = session['user_id']
        stock_id = int(request.form['stock_id'])
        target_price = Decimal(request.form['target_price'])
        action = request.form['action']
        quantity = int(request.form['quantity'])

        # Validate quantity
        if quantity <= 0:
            flash("Quantity must be greater than 0.", "danger")
            return redirect(url_for('stop_loss'))

        # Additional validation for sell orders
        if action == 'sell':
            portfolio = Portfolio.query.filter_by(user_id=user_id, stock_id=stock_id).first()
            if not portfolio or portfolio.quantity < quantity:
                flash("Insufficient shares to sell.", "danger")
                return redirect(url_for('stop_loss'))

        # Additional validation for buy orders
        if action == 'buy':
            stock = Stocks.query.get(stock_id)
            if stock:
                total_cost = Decimal(stock.current_price) * quantity
                funds = Funds.query.filter_by(user_id=user_id).first()
                if not funds or funds.available_balance < float(total_cost):
                    flash("Insufficient funds for this order.", "danger")
                    return redirect(url_for('stop_loss'))

        # Start monitoring in a separate thread
        thread = Thread(
            target=monitor_stock_price,
            args=(user_id, stock_id, target_price, action, quantity)
        )
        thread.daemon = True
        thread.start()

        flash(f"Stop-loss monitoring started for {quantity} shares.", "info")
        return redirect(url_for('portfolio'))

    # Get list of stocks and user's portfolio for the template
    stocks = Stocks.query.all()
    portfolio = Portfolio.query.filter_by(user_id=session['user_id']).all()
    
    return render_template('stock_loss.html', stocks=stocks, portfolio=portfolio)
if __name__ == '__main__':
    start_stock_threads()
    with app.app_context():
        db.create_all()
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=80)