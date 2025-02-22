from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os
import uuid
import base64
import requests
import re
import xml.etree.ElementTree as ET
import io
import datetime
from collections import Counter

# AI & NLP Libraries
import openai
import nltk
import spacy
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Graph & Visualization Libraries
import networkx as nx
import pyvis
import graphviz
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# PDF Handling
from PyPDF2 import PdfReader

# Database
from sqlalchemy.sql import text

# Custom Module
from citation_generator import CitationGenerator




# Manually set the Graphviz executable path (adjust this to match your installation path)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for the app

API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"  # Semantic Scholar API
# Define API endpoints for Semantic Scholar and ArXiv
SEMANTIC_API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_ENDPOINT = "http://export.arxiv.org/api/query"


# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

CORS(app)  # Enable CORS for all routes

# Configure SQLAlchemy for PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Gownahalli123!@localhost/research_assistant'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ✅ Load NLP Models (Only Once)
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the Hugging Face BART model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/research-tutorials')
def research_tutorials():
    # You can replace "Researcher" with dynamic data from your session or database.
    user = "Researcher"
    return render_template('research_tutorials.html', user=user)

@app.route('/summarizer', methods=['GET', 'POST'])
def summarizer():
    try:
        if request.method == 'POST':
            if not request.is_json:
                return jsonify({"error": "Unsupported Media Type. Use 'application/json'"}), 415

            data = request.get_json()
            text = data.get("text", "").strip()

            if not text:
                return jsonify({"error": "No text provided"}), 400

            # ✅ Softly increase max_length while ensuring min_length is reasonable
            max_length = min(len(text.split()) * 0.75, 250)
            min_length = max(int(len(text.split()) * 0.4), 30)

            summary = abstractive_summary(text, max_length, min_length)

            return jsonify({"summary": summary})

        return render_template('summarizer.html')

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def summarize_text(text):
    """Perform hybrid summarization (extractive + abstractive)."""
    extractive = extractive_summary(text, num_sentences=3)  # Step 1: Extract key sentences
    abstractive = abstractive_summary(extractive)  # Step 2: Rewrite it using the model
    return abstractive  # Return final improved summary

def extractive_summary(text, num_sentences=3):
    """Extract key sentences based on importance."""
    doc = nlp(text)
    sentences = sent_tokenize(text)

    # Score sentences based on named entity presence & length
    sentence_scores = {sent: len(sent.split()) + len([ent for ent in doc.ents if ent.text in sent]) for sent in sentences}
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    extractive_summary = " ".join(sorted_sentences[:num_sentences])
    return clean_text(extractive_summary)

def abstractive_summary(text, max_length, min_length):
    """Generate a high-quality abstractive summary using T5."""
    try:
        if len(text.split()) < 20:
            return text  # If input is too short, return as-is

        input_text = f"summarize: {text}"
        print(f"🔹 Input Text for Model: {input_text}")

        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )

        # ✅ Move model & inputs to CPU for stability
        model.to("cpu")
        inputs = inputs.to("cpu")

        # ✅ Ensure integer min/max length (from user input)
        max_length = int(max_length)
        min_length = int(min_length)

        # ✅ Generate summary with enhanced settings
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=max_length,  
            min_length=min_length,  
            length_penalty=1.0,  # Balances conciseness and details
            num_beams=8,  # Increases quality by considering more variations
            repetition_penalty=2.0,  # Prevents repeating words
            temperature=0.9,  # Allows more diverse word choices
            do_sample=True,  # ✅ Enables temperature-based sampling
            early_stopping=True  # Stops generation at a natural ending
        )

        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # ✅ Ensure the summary ends properly
        if not summarized_text.endswith((".", "!", "?")):
            summarized_text += "."

        # ✅ Debugging Info
        print(f"✅ Generated Summary: {summarized_text}")
        print(f"✅ Raw Output Tokens: {summary_ids}")

        # ✅ If summary is too short, warn user
        if len(summarized_text.split()) < min_length:
            summarized_text += " (⚠️ Summary is shorter than expected, consider increasing max_length)."

        return summarized_text.strip()

    except Exception as e:
        print(f"⚠️ Error in abstractive_summary: {e}")
        return "Error generating summary."



def clean_text(text):
    """Preprocess text by removing extra spaces."""
    return re.sub(r'\s+', ' ', text).strip()

# Define database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Document(db.Model):
    __tablename__ = 'Document'  # Ensure this matches the table name in your database

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('documents', lazy=True))

class SavedArticle(db.Model):
    __tablename__ = 'saved_articles'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    authors = db.Column(db.String(255), nullable=True)
    year = db.Column(db.String(10), nullable=True)
    venue = db.Column(db.String(255), nullable=True)
    url = db.Column(db.String(500), nullable=False)

    user = db.relationship('User', backref=db.backref('saved_articles', lazy=True))




# Initialize the database within the app context
with app.app_context():
    try:
        db.create_all()  # Ensure tables are created
        db.session.execute(text('SELECT 1'))  # Test the connection
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")

# Utility Functions
def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

@app.route('/')
def index():
    return render_template('index.html')  # Replace with your homepage template


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page where users can log into their account."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return render_template('login.html', error="Both username and password are required.")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id  # Store user ID in session
            session['user'] = username   # Store username in session
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page where new users can create an account."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template('register.html', error="All fields are required.")

        # Check for existing username or email
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists.")
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already exists.")

        # Create a new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard to display user's documents and allow creating new ones."""
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    user = User.query.filter_by(username=username).first()

    if not user:
        return redirect(url_for('login'))

    # Fetch documents associated with the user, ordered by the most recent update
    documents = Document.query.filter_by(user_id=user.id).order_by(Document.updated_at.desc()).all()

    return render_template('dashboard.html', user=username, documents=documents)







@app.route('/new-document', methods=['GET', 'POST'])
def new_document():
    """Create a new document."""
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    user = User.query.filter_by(username=username).first()

    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')

        new_doc = Document(user_id=user.id, title=title, content=content)
        db.session.add(new_doc)
        db.session.commit()

        return redirect(url_for('dashboard'))
    return render_template('new_document.html')

@app.route('/save-document', methods=['POST'])
def save_document():
    """Save document to the database."""
    if 'user' not in session:
        return jsonify({"status": "error", "message": "You must be logged in to save documents"}), 401

    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()

        if not title or not content:
            return jsonify({"status": "error", "message": "Both title and content are required"}), 400

        # Save to database (Example with SQLAlchemy)
        new_document = Document(
            user_id=session['user_id'],
            title=title,
            content=content,
            created_at=datetime.now()
        )
        db.session.add(new_document)
        db.session.commit()

        return jsonify({"status": "success", "message": "Document saved successfully"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error saving document: {e}"}), 500



@app.route('/trend-analyzer', methods=['GET', 'POST'])
def trend_analyzer():
    if request.method == 'POST':  # Handle AJAX request for analysis
        text_input = request.form.get('text_input')
        file = request.files.get('file')
        result = None

        if text_input:
            result = analyze_text_professional(text_input)  # AI-based text analysis
        elif file:
            try:
                df, stats, correlation_matrix = analyze_data(file)
                result = {"stats": stats.to_dict(), "correlation": correlation_matrix.to_dict()}
            except Exception as e:
                result = {"error": f"Error processing file: {e}"}
        else:
            result = {"error": "No text or file provided for analysis."}

        return jsonify(result)  # Return analysis data for AJAX

    # Render the template when loading the page (GET request)
    return render_template('trend_analyzer.html')


def analyze_text_professional(input_text):
    """
    Uses a T5 model to generate a summary/hypothesis.
    You can adjust the parameters (like max_length) as needed.
    """
    # Prepend the task to the input (T5 expects a prefix for its tasks)
    input_text_with_prefix = "summarize: " + input_text.strip()
    
    # Tokenize and encode the input text
    input_ids = tokenizer.encode(input_text_with_prefix, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a summary using the model
    summary_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated summary
    hypothesis = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"hypothesis": hypothesis}



### 🔹 FILE ANALYSIS FUNCTION ###
def analyze_data(file):
    """Processes uploaded CSV file for insights."""
    df = pd.read_csv(file)  # Ensure the file is CSV format
    stats = df.describe()
    correlation_matrix = df.corr()
    return df, stats, correlation_matrix


@app.route('/generate-chart', methods=['POST'])
def generate_chart_api():
    try:
        x_labels = request.form.getlist("x_labels[]")
        y_values = list(map(float, request.form.getlist("numbers[]")))
        chart_type = request.form.get("chart_type")
        x_axis_label = request.form.get("x_axis_label", "X-axis")
        y_axis_label = request.form.get("y_axis_label", "Y-axis")

        print("Chart Type:", chart_type)
        print("X Labels:", x_labels)
        print("Y Values:", y_values)

        df = pd.DataFrame({"X": x_labels, "Y": y_values})
        chart_url = generate_chart(df, chart_type, x_axis_label, y_axis_label)

        return jsonify({"chart_url": chart_url})
    except Exception as e:
        return jsonify({"error": str(e)})



def generate_chart(df, chart_type, x_label, y_label):
    plt.figure(figsize=(6, 4))

    if chart_type == "line":
        plt.plot(df["X"], df["Y"], marker="o", linestyle="-", color="b")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Line Graph")
    elif chart_type == "bar":
        plt.bar(df["X"], df["Y"], color="g")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Bar Chart")
    elif chart_type == "pie":
        # Define a longer list of colors if needed
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ffb3e6", "#c2d6d6"]
        plt.pie(df["Y"], labels=df["X"], autopct="%1.1f%%", colors=colors[:len(df)])
        plt.title("Pie Chart")

        # Pie charts generally don't have x and y axis labels.
    
    plt.xticks(rotation=30)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


### 🔹 ARTICLE FINDER FUNCTION ###
@app.route('/article-finder', methods=['GET', 'POST'])
def article_finder():
    results = []
    error = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if not query:
            error = "Please enter a search query."
        else:
            try:
                # Fetch from Semantic Scholar
                semantic_params = {
                    'query': query,
                    'limit': 5,
                    'fields': 'title,authors,year,venue,url',
                }
                semantic_response = requests.get(SEMANTIC_API_ENDPOINT, params=semantic_params)
                if semantic_response.status_code == 200:
                    semantic_data = semantic_response.json()
                    for paper in semantic_data.get('data', []):
                        results.append({
                            'title': paper.get('title', 'No title available'),
                            'authors': [author.get('name', '') for author in paper.get('authors', [])],
                            'year': paper.get('year', 'Unknown year'),
                            'venue': paper.get('venue', 'Unknown venue'),
                            'url': paper.get('url', '#'),
                            'source': 'Semantic Scholar'
                        })
                else:
                    print(f"Error fetching from Semantic Scholar: {semantic_response.status_code}")

                # Fetch from ArXiv
                arxiv_params = {
                    'search_query': f'all:{query}',
                    'start': 0,
                    'max_results': 5,
                }
                arxiv_response = requests.get(ARXIV_API_ENDPOINT, params=arxiv_params)
                if arxiv_response.status_code == 200:
                    root = ET.fromstring(arxiv_response.content)
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        try:
                            title = entry.find('{http://www.w3.org/2005/Atom}title')
                            title_text = title.text if title is not None else "No title available"

                            link = entry.find('{http://www.w3.org/2005/Atom}id')
                            link_text = link.text if link is not None else "#"

                            authors = [
                                author.find('{http://www.w3.org/2005/Atom}name').text
                                for author in entry.findall('{http://www.w3.org/2005/Atom}author')
                                if author.find('{http://www.w3.org/2005/Atom}name') is not None
                            ]

                            published = entry.find('{http://arxiv.org/schemas/atom}published')
                            year = published.text[:4] if published is not None else "Unknown year"

                            results.append({
                                'title': title_text,
                                'authors': authors,
                                'year': year,
                                'venue': 'ArXiv',
                                'url': link_text,
                                'source': 'ArXiv'
                            })
                        except Exception as e:
                            print(f"Error parsing ArXiv entry: {e}")
                else:
                    print(f"Error fetching from ArXiv: {arxiv_response.status_code}")
            except Exception as e:
                error = f"Error during search: {e}"

    return render_template('article_finder.html', results=results, error=error)






@app.route('/save-article', methods=['POST'])
def save_article():
    if 'user' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    try:
        data = request.json
        user_id = session['user_id']  # Assuming session stores user ID
        saved_article = SavedArticle(
            user_id=user_id,
            title=data['title'],
            authors=data['authors'],
            year=data['year'],
            venue=data['venue'],
            url=data['url']
        )
        db.session.add(saved_article)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Article saved successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error saving article: {e}'})


@app.route('/library', methods=['GET', 'POST'])
def library():
    if request.method == 'POST':
        # Delete the article by ID
        article_id = int(request.form.get('article_id'))
        article = SavedArticle.query.get(article_id)
        if article:
            db.session.delete(article)
            db.session.commit()
        return redirect(url_for('library'))

    # Fetch saved articles
    saved_articles = SavedArticle.query.all()
    return render_template('library.html', saved_articles=saved_articles)



class CitationGenerator:
    def __init__(self, authors, title, publisher, year, url="", volume="", pages=""):
        # Expect authors to be a list; remove any empty strings and strip whitespace.
        self.authors = [a.strip() for a in authors if a.strip()]
        self.title = title.strip()
        # Publisher is optional.
        self.publisher = publisher.strip() if publisher.strip() else ""
        self.year = year.strip()
        self.url = url.strip()
        self.volume = volume.strip()
        self.pages = pages.strip()

    def format_author_apa_single(self, author):
        """Format a single author name for APA as 'Last, F.'"""
        parts = author.split()
        if len(parts) > 1:
            last_name = parts[-1]
            first_initial = parts[0][0] + "."
            return f"{last_name}, {first_initial}"
        return author

    def format_authors_apa(self):
        """Format multiple authors for APA style."""
        if not self.authors:
            return ""
        if len(self.authors) == 1:
            return self.format_author_apa_single(self.authors[0])
        formatted_list = [self.format_author_apa_single(author) for author in self.authors]
        return ", ".join(formatted_list[:-1]) + " & " + formatted_list[-1]

    def format_authors_mla_chicago(self):
        """Format multiple authors for MLA/Chicago style."""
        if not self.authors:
            return ""
        if len(self.authors) == 1:
            return self.authors[0]
        return ", ".join(self.authors[:-1]) + " and " + self.authors[-1]

    def generate(self, format):
        if format.lower() == "apa":
            authors_formatted = self.format_authors_apa()
            citation = f"{authors_formatted} ({self.year}). <i>{self.title}</i>."
            # Build a list of details to include after the title.
            details = []
            if self.publisher:
                details.append(self.publisher)
            # Even if publisher is not provided, include volume and pages if available.
            if self.volume:
                details.append(self.volume)
            if self.pages:
                details.append(self.pages)
            # If any details exist, join them with commas and append.
            if details:
                citation += " " + ", ".join(details) + "."
            if self.url:
                citation += f" Retrieved from {self.url}"
            return citation

        elif format.lower() == "mla":
            authors_formatted = self.format_authors_mla_chicago()
            citation = f"{authors_formatted}. <i>{self.title}</i>."
            if self.publisher:
                citation += f" {self.publisher},"
            citation += f" {self.year}."
            if self.url:
                citation += f" {self.url}"
            return citation

        elif format.lower() == "chicago":
            authors_formatted = self.format_authors_mla_chicago()
            citation = f"{authors_formatted}. <i>{self.title}</i>."
            if self.publisher:
                citation += f" {self.publisher},"
            citation += f" {self.year}."
            if self.url:
                citation += f" Accessed {self.url}."
            return citation

        else:
            return "Unsupported format"




@app.route('/citation-generator', methods=['GET', 'POST'])
def citation_generator():
    """Generate citations in selected format and build bibliography."""
    if 'user' not in session:
        return redirect(url_for('login'))  # Ensure the user is logged in

    bibliography = None
    error = None

    # Initialize session citations if not already present
    if 'citations' not in session:
        session['citations'] = []

    if request.method == 'POST' and 'format' in request.form:
        # Retrieve multiple authors using getlist (the HTML input name should be "author[]")
        authors = request.form.getlist('author[]')
        title = request.form.get('title').strip()
        publisher = request.form.get('publisher', '').strip()
        year = request.form.get('year').strip()
        url = request.form.get('url', '').strip()
        volume = request.form.get('volume', '').strip()
        pages = request.form.get('pages', '').strip()
        citation_format = request.form.get('format')

        # Only require authors, title, and year
        if not authors or not title or not year:
            error = "Please fill in all the required fields (authors, title, and year are required)."
        else:
            try:
                # Generate citation using the updated CitationGenerator class
                citation_gen = CitationGenerator(authors, title, publisher, year, url, volume, pages)
                citation = citation_gen.generate(citation_format)

                if citation == "Unsupported format":
                    error = "Selected citation format is unsupported."
                else:
                    # Save citation to session along with authors (for sorting)
                    session['citations'].append({
                        "authors": authors,
                        "citation": citation
                    })
                    session.modified = True

            except Exception as e:
                error = f"Error generating citation: {e}"

    # Sort citations alphabetically by the first author's name
    if session['citations']:
        sorted_citations = sorted(session['citations'], key=lambda x: x['authors'][0].lower())
        session['citations'] = sorted_citations
        bibliography = "\n".join([c['citation'] for c in sorted_citations])

    return render_template('citation_generator.html', error=error, bibliography=bibliography)



@app.route('/remove-citation', methods=['POST'])
def remove_citation():
    """Remove a specific citation from the bibliography."""
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'citations' in session:
        index = int(request.form.get('index'))
        if 0 <= index < len(session['citations']):
            session['citations'].pop(index)
            session.modified = True

    return redirect(url_for('citation_generator'))





def generate_mermaid_diagram(root, branches):
    """
    Generate a Mermaid.js diagram definition.
    Args:
        root (str): The central concept.
        branches (dict): A dictionary of branches and sub-branches.
    Returns:
        str: Mermaid.js diagram definition.
    """
    try:
        diagram = f"graph TD\n    A[{root}]\n"
        for branch, sub_branches in branches.items():
            branch_id = branch.replace(" ", "_")
            diagram += f"    A --> {branch_id}[{branch}]\n"
            for sub in sub_branches:
                sub_id = sub.replace(" ", "_")
                diagram += f"    {branch_id} --> {sub_id}[{sub}]\n"
        return diagram
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        return ""


# Update the mindmap function to use the enhanced AI functionality
@app.route('/mindmap', methods=['GET', 'POST'])
def mindmap():
    mindmap_diagram = None
    error = None

    if request.method == 'POST':
        keywords = request.form.get('keywords')
        if not keywords:
            error = "Please enter some keywords to generate the mind map."
        else:
            try:
                # Process keywords into a dictionary
                root_concept = "Central Idea"
                branches = {}
                for keyword in keywords.split(','):
                    branch, *sub_branches = keyword.split(':')
                    branches[branch.strip()] = [sub.strip() for sub in sub_branches if sub.strip()]

                # Generate Mermaid.js diagram
                mindmap_diagram = generate_mermaid_diagram(root_concept, branches)
            except Exception as e:
                error = f"Error generating mind map: {e}"

    return render_template('mindmap.html', mindmap_diagram=mindmap_diagram, error=error)


def create_mindmap(root, branches, output_file='mindmap', file_format='png'):
    """
    Create a visually appealing mind map with AI-enhanced branches.
    """
    try:
        # Create a Digraph with enhanced visuals
        dot = Digraph(comment="Mind Map", graph_attr={'splines': 'true'})
        dot.node("Root", root, shape='ellipse', style='filled', color='lightblue', fontcolor='black')

        # Add branches and sub-branches with styles
        for branch, sub_branches in branches.items():
            dot.node(branch, branch, shape='box', style='filled', color='lightgreen', fontcolor='black')
            dot.edge("Root", branch)
            for sub in sub_branches:
                dot.node(sub, sub, shape='circle', style='filled', color='lightyellow', fontcolor='black')
                dot.edge(branch, sub)

        # Render the mind map and save it
        output_path = f'static/{output_file}.{file_format}'
        dot.render(output_path, format=file_format, cleanup=True)
        return output_path
    except Exception as e:
        print(f"Error generating mind map: {e}")
        return None



# Define the To-Do Model
class ToDo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    deadline = db.Column(db.String(20), nullable=True)
    priority = db.Column(db.String(10), nullable=False)
    done = db.Column(db.Boolean, default=False)

# Initialize the database within the app context
with app.app_context():
    try:
        db.create_all()  # Ensure tables are created
        db.session.execute(text('SELECT 1'))  # Test the connection
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")

@app.route('/todo-list')
def todo_list():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('todo_list.html')


# API Routes for To-Do List
@app.route('/tasks', methods=['GET'])
def get_tasks():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    tasks = ToDo.query.filter_by(user_id=session['user_id']).all()
    return jsonify([{
        'id': task.id,
        'text': task.text,
        'category': task.category,
        'deadline': task.deadline,
        'priority': task.priority,
        'done': task.done
    } for task in tasks])

@app.route('/tasks', methods=['POST'])
def add_task():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    # ✅ Ensure request is JSON
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON format'}), 400

    data = request.get_json()  # ✅ Properly parse JSON data

    try:
        new_task = ToDo(
            user_id=session['user_id'],
            text=data['text'],
            category=data['category'],
            deadline=data.get('deadline', ''),
            priority=data['priority'],
            done=False
        )
        db.session.add(new_task)
        db.session.commit()
        return jsonify({'message': 'Task added successfully'}), 201

    except Exception as e:
        return jsonify({'error': f'Failed to add task: {str(e)}'}), 500


@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = ToDo.query.get(task_id)
    if not task or task.user_id != session.get('user_id'):
        return jsonify({'error': 'Task not found'}), 404
    data = request.json
    task.text = data.get('text', task.text)
    task.category = data.get('category', task.category)
    task.deadline = data.get('deadline', task.deadline)
    task.priority = data.get('priority', task.priority)
    task.done = data.get('done', task.done)
    db.session.commit()
    return jsonify({'message': 'Task updated successfully'})

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = ToDo.query.get(task_id)
    if not task or task.user_id != session.get('user_id'):
        return jsonify({'error': 'Task not found'}), 404
    db.session.delete(task)
    db.session.commit()
    return jsonify({'message': 'Task deleted successfully'})


if __name__ == "__main__":
    # Ensure the database tables are created before starting the app
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)

