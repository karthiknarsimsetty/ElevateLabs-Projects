from flask import Flask, render_template, request, send_file
from utils import (
    extract_text_from_pdf,
    preprocess_text,
    calculate_similarity,
    get_skill_match,
    extract_name,
    extract_email,
    extract_phone
)

import os
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/rank", methods=["POST"])
def rank():

    resumes = request.files.getlist("resumes")

    job_description = preprocess_text(
        request.form["job_description"]
    )

    results = []

    for resume in resumes:

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            resume.filename
        )

        resume.save(filepath)

        raw_text = extract_text_from_pdf(filepath)

        name = extract_name(raw_text)
        email = extract_email(raw_text)
        phone = extract_phone(raw_text)

        cleaned_text = preprocess_text(raw_text)

        score = calculate_similarity(
            job_description,
            [cleaned_text]
        )[0]

        matched, missing = get_skill_match(
            job_description,
            cleaned_text
        )

        if score >= 0.30:
            recommendation = "Excellent Match"
        elif score >= 0.20:
            recommendation = "Good Match"
        elif score >= 0.10:
            recommendation = "Average Match"
        else:
            recommendation = "Poor Match"

        results.append({

            "name": name,

            "email": email,

            "phone": phone,

            "filename": resume.filename,

            "score": round(score * 100, 2),

            "matched": ", ".join(matched),

            "missing": ", ".join(missing),

            "recommendation": recommendation

        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    df = pd.DataFrame(results)

    df.insert(0, "Rank", range(1, len(df) + 1))

    df.to_csv(
        "reports/resume_ranking.csv",
        index=False
    )

    return render_template(
        "results.html",
        results=results
    )


@app.route("/download")
def download():

    return send_file(
        "reports/resume_ranking.csv",
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)