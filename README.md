<img width="1897" height="913" alt="Screenshot 2026-07-13 225104" src="https://github.com/user-attachments/assets/26e49866-1181-4fb4-a3fa-b68620f838b0" />
![Uploading image.png…]()

# AI Resume Analyzer & Job Description Matcher

An AI-powered web application built with **FastAPI** and **Machine Learning** that parses resumes (PDFs), extracts core technical skills, and matches them against job descriptions. The system leverages natural language processing (NLP) and a trained **Logistic Regression** classifier to predict the target job role and calculate a confidence match score.

## 🚀 Key Features
- **PDF Text Parsing**: Automated text extraction from uploaded resume files using `pdfplumber`.
- **Advanced Text Preprocessing**: Cleans and normalizes text tokens while handling special technical characters (like `C++`, `#`, `.net`).
- **Robust Skill Extraction**: Matches multi-word domains and hyphenated phrases (e.g., "Full Stack Web Development", "React JS") seamlessly by normalizing whitespace and casing.
- **Machine Learning Role Prediction**: Utilizes a TF-IDF vectorizer paired with a Logistic Regression model to accurately classify resume backgrounds into technical roles.
- **Confidence Metrics**: Outputs a structured breakdown of the top 5 predicted roles with percentage confidence scores.

---

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI**: Modern, high-performance web framework for building APIs with Python.
- **Uvicorn**: Lightning-fast ASGI server implementation for production hosting.
- **Jinja2**: Templating engine used for rendering the frontend user interface (`ui.html`).

### Machine Learning & NLP
- **scikit-learn (v1.8.0)**: Powers the TF-IDF feature extraction and Logistic Regression classification.
- **NLTK**: Used to access and filter out English stop words during the text preprocessing layer.
- **Joblib**: For fast and optimized serialization/deserialization of the trained model and vectorizer.
- **pdfplumber**: Handles structural text layout extraction from PDF files.

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**: 
   Text from resumes and prompts is converted to lowercase. Stop words are stripped away using NLTK datasets. Tokenization isolates alphanumeric phrases along with specific symbols using custom Regular Expressions (`[a-zA-Z0-9+#.-]+`).
2. **Feature Engineering**: 
   The cleaned text is processed using **TF-IDF Vectorization** (`TfidfVectorizer`). This translates raw text strings into numerical feature matrices based on term importance across your training dataset.
3. **Classification Model**: 
   A **Logistic Regression** classifier learns from the TF-IDF representation of resumes to predict categorical job titles. When a user queries the app, the model provides both a discrete class prediction and an array of probability metrics (`predict_proba`) representing match confidence.

---

## 💻 How to Run Locally

Follow these steps to set up and run the application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd AIResume-Analyzer
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Ensure you install the exact version pins required for environment stability:
```bash
pip install -r requirements.txt
```

### 4. Place Your Model Files
Ensure your root directory contains your serialized model and data files:
- `logestic_regression.joblib`
- `vectorizer.joblib`
- `data.py` (containing your `skilled_db`)

### 5. Start the Application
Run the local production reloader via python:
```bash
python main.py
```
Your application will be live locally at: **`http://127.0.0.1:8081`**

---

## 🌐 Production Deployment (Render)

This project is fully configured for zero-downtime deployment on **Render**.

### Render Web Service Settings:
When configuring your new Web Service on the Render Dashboard, use the following operational parameters:

- **Environment/Language**: `Python`
- **Region**: Select your closest region
- **Branch**: `main`
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port \$PORT
  ```

*Note: The project imports `warnings` to automatically handle scikit-learn environment matching warnings cleanly in Render cloud production environments.*
