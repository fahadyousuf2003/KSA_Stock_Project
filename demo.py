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
import json
import pathlib
import textwrap
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from datetime import date

from IPython.display import display
from IPython.display import Markdown
# Import Libraries
import os
import pandas as pd
import tiktoken

from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

import io
from contextlib import redirect_stdout

nlp = spacy.load("en_core_web_sm")
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://root:@localhost/KSA_Stock_Project'
db = SQLAlchemy(app)
groq_api_key = "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"
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
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False) 
    quantity = db.Column(db.Integer, nullable=False)
    purchase_price = db.Column(DECIMAL(10,2), nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)

class Funds(db.Model):
    __tablename__ = 'funds'

    fund_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_info.id'), nullable=False)
    available_balance = db.Column(db.Float, nullable=False, default=0.0)
    total_balance = db.Column(db.Float, nullable=False, default=0.0)
    last_updated = db.Column(db.DateTime, nullable=False)

from sqlalchemy.orm import joinedload
import pandas as pd

# Perform the join
with app.app_context():
    query = (
        db.session.query(
            User_info.id.label("user_id"),
            User_info.name.label("user_name"),
            User_info.Surname.label("user_surname"),
            User_info.Gender.label("user_gender"),
            Stocks.stock_id.label("stock_id"),
            Stocks.ticker_symbol.label("ticker_symbol"),
            Stocks.company_name.label("company_name"),
            Stocks.current_price.label("current_price"),
            Stocks.sector.label("sector"),
            StockPriceHistory.price.label("stock_price"),
            StockPriceHistory.date.label("price_date"),
            Portfolio.quantity.label("quantity"),
            Portfolio.purchase_price.label("purchase_price"),
            Portfolio.purchase_date.label("purchase_date"),
            Funds.available_balance.label("available_balance"),
            Funds.total_balance.label("total_balance"),
            Funds.last_updated.label("fund_last_updated"),
        )
        .join(Portfolio, User_info.id == Portfolio.user_id)
        .join(Stocks, Portfolio.stock_id == Stocks.stock_id)
        .join(StockPriceHistory, Stocks.stock_id == StockPriceHistory.stock_id)
        .join(Funds, User_info.id == Funds.user_id)
    )

    # Fetch all results
    results = query.all()

    # Convert query results to a list of dictionaries
    data = [dict(row._mapping) for row in results]

    # Create a Pandas DataFrame
    combined_data = pd.DataFrame(data)

    # Save the combined data to a CSV file
    combined_data.to_csv('combined_data.csv', index=False)
    print("Combined data has been successfully saved to 'combined_data.csv'!")

# Create combined information column
combined_data['combined_info'] = combined_data.apply(
    lambda row: (
        f"User: {row['user_name']} {row['user_surname']} (Gender: {row['user_gender']}). "
        f"Stock: {row['ticker_symbol']} - {row['company_name']} in {row['sector']} sector, "
        f"current price: {row['current_price']}. "
        f"Portfolio: {row['quantity']} shares purchased at {row['purchase_price']} on {row['purchase_date']}. "
        f"Stock Price History: {row['stock_price']} on {row['price_date']}. "
        f"Funds: Available balance: {row['available_balance']}, Total balance: {row['total_balance']} (Last updated: {row['fund_last_updated']})."
    ),
    axis=1
)
# Token encoding
encoding = tiktoken.get_encoding('cl100k_base')
max_tokens = 8000

# Omit descriptions that are too long to embed
combined_data["n_tokens"] = combined_data.combined_info.apply(lambda x: len(encoding.encode(x)))
combined_data = combined_data[combined_data.n_tokens <= max_tokens]

# Use HuggingFace embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
combined_data["vector"] = combined_data.combined_info.apply(lambda x: embeddings.embed_query(x))
combined_data.to_pickle('combined_data.pkl')
# Set the API key as an environment variable
os.environ["PINECONE_API_KEY"] = "pcsk_7YQanE_AP9y5db9N5vQaoUYC2h6bxvr92sEPyXzVBUcotcvBubtsEqsfkDaLQ6LrGwWRnw"

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
documents = [
    Document(
        page_content=row['combined_info'],
        metadata={
            'user_id': row['user_id'],
        }
    ) for _, row in combined_data.iterrows()
]
index_name = "combineddataindex"
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
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

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

def retrieve_ranked_context(user_id, query):
    # Filter the documents based on user_id
    results = docsearch.similarity_search(query, k=100)  # Retrieve top 100 matches
    
    # Filter results to only include those related to the given user_id
    user_specific_results = [ doc for doc in results if str(doc.metadata.get('user_id')) == str(user_id)]
    if not user_specific_results:
        return "Sorry, I couldn't find any specific information for this user. Here's some general information about stocks and portfolios."
    
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

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def get_stock_info(user_id, query):
    # Suppress any print outputs during the execution
    with io.StringIO() as buf, redirect_stdout(buf):
        # Retrieve context specific to the user
        context = retrieve_ranked_context(user_id, query)
        qa_input = {"context": context, "query": query}
        result = qa_chain(qa_input)

    # Return the final result without any intermediate prints
    return result['result']
import tensorflow as tf

# Using the newer API
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

user_id = 1 # Example user_id
query =  "what is my name"  # Example query
result = get_stock_info(user_id, query)
print(result)
