import io
import joblib
import pdfplumber
import uvicorn
import nltk
import re
import warnings

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from nltk.corpus import stopwords
from sklearn.exceptions import InconsistentVersionWarning

from data import skilled_db

# ==========================
# Safety & Warnings
# ==========================
# Suppress the version warning logs on Render if versions are mismatched
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==========================
# Load Model
# ==========================
model = joblib.load("logestic_regression.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# NLTK Initialization
# ==========================
# "punkt" was removed since it is completely unused by your custom regex tokeniser
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ==========================
# Home Page
# ==========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})

# ==========================
# PDF Text Extraction
# ==========================
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ==========================
# Text Preprocessing
# ==========================
def preprocess(text):
    text = text.lower()
    # FIX 1: Added a hyphen (-) to the regex so phrases like "full-stack" do not get broken apart
    tokens = re.findall(r"[a-zA-Z0-9+#.-]+", text)
    
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# ==========================
# Skill Extraction
# ==========================
def extract_skills(text):
    found_skills = set()
    text = text.lower()

    for category, skills in skilled_db.items():
        for skill in skills:
            # FIX 2: Normalized internal spacing and hyphens within target skills 
            clean_skill = skill.lower().strip()
            clean_skill = re.sub(r"[-_\s]+", " ", clean_skill)
            
            # Normalize target text spacing for accurate phrase matching
            normalized_text = re.sub(r"[-_\s]+", " ", text)

            pattern = r"\b" + re.escape(clean_skill) + r"\b"
            if re.search(pattern, normalized_text):
                found_skills.add(skill)

    return sorted(list(found_skills))

# ==========================
# Upload API
# ==========================
@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form("")
):
    contents = await file.read()
    resume_text = extract_text(io.BytesIO(contents))

    clean_resume = preprocess(resume_text)
    resume_skills = extract_skills(clean_resume)

    clean_jd = preprocess(job_description)
    jd_skills = extract_skills(clean_jd)

    missing_skills = list(set(jd_skills) - set(resume_skills))
    suggestions = [f"Learn {skill}" for skill in missing_skills]

    # FIX 3: Classification logic isolation
    # If your model was trained strictly on resumes to predict a job role, 
    # blending the JD into the vector corrupts the math. We classify the resume directly.
    vector = vectorizer.transform([clean_resume])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    role_confidence = sorted(
        [
            {
                "role": role,
                "confidence": round(score * 100, 2)
            }
            for role, score in zip(model.classes_, probabilities)
        ],
        key=lambda x: x["confidence"],
        reverse=True
    )

    return {
        "predicted_role": prediction,
        "skills_found": resume_skills,
        "total_skills_found": len(resume_skills),
        "job_required_skills": jd_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "role_confidence": role_confidence[:5]
    }

# ==========================
# Local Run
# ==========================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8081,
        reload=True
    )
