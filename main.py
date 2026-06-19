import io
import joblib
import pdfplumber
import uvicorn
import nltk
import re

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from data import skilled_db


# ==========================
# Load Model
# ==========================

model = joblib.load("logestic_regression.joblib")
vectorizer = joblib.load("vectorizer.joblib")


# ==========================
# FastAPI
# ==========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# NLTK
# ==========================

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ==========================
# Extract PDF Text
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
# Preprocessing
# ==========================

def preprocess(text):

    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z0-9+#.]+",
        text
    )

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(tokens)


# ==========================
# Skill Extraction
# ==========================

def extract_skills(text):

    found_skills = set()

    text = text.lower()

    for category, skills in skilled_db.items():

        for skill in skills:

            skill_lower = skill.lower()

            pattern = r"\b" + re.escape(skill_lower) + r"\b"

            if re.search(pattern, text):
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

    try:

        contents = await file.read()

        # -------------------
        # Resume Text
        # -------------------

        resume_text = extract_text(
            io.BytesIO(contents)
        )

        clean_resume = preprocess(
            resume_text
        )

        # -------------------
        # Resume Skills
        # -------------------

        resume_skills = extract_skills(
            clean_resume
        )

        # -------------------
        # JD Skills
        # -------------------

        clean_jd = preprocess(
            job_description
        )

        jd_skills = extract_skills(
            clean_jd
        )

        # -------------------
        # Missing Skills
        # -------------------

        missing_skills = list(
            set(jd_skills) - set(resume_skills)
        )

        suggestions = [
            f"Learn {skill}"
            for skill in missing_skills
        ]

        # -------------------
        # Prediction
        # Resume + JD
        # -------------------

        combined_text = (
            clean_resume +
            " " +
            clean_jd
        )

        vector = vectorizer.transform(
            [combined_text]
        )

        prediction = model.predict(
            vector
        )[0]

        probabilities = model.predict_proba(
            vector
        )[0]

        role_confidence = sorted(
            [
                {
                    "role": role,
                    "confidence": round(
                        score * 100,
                        2
                    )
                }
                for role, score in zip(
                    model.classes_,
                    probabilities
                )
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )

        # Top 5 only
        role_confidence = role_confidence[:5]

        return {

            "predicted_role": prediction,

            "skills_found": resume_skills,

            "total_skills_found": len(
                resume_skills
            ),

            "job_required_skills": jd_skills,

            "missing_skills": missing_skills,

            "suggestions": suggestions,

            "role_confidence": role_confidence
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ==========================
# Run
# ==========================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8081,
        reload=True
    )