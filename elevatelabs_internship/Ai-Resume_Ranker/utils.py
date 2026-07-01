import fitz
import spacy
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


def preprocess_text(text):

    doc = nlp(text.lower())

    words = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.is_space:
            continue

        words.append(token.lemma_)

    return " ".join(words)


def calculate_similarity(job_description, resumes):

    documents = [job_description] + resumes

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf[0:1],
        tfidf[1:]
    ).flatten()

    return similarity


def load_skills():

    with open("skills.txt", "r") as file:
        skills = [skill.strip().lower() for skill in file.readlines()]

    return skills


def get_skill_match(job_description, resume_text):

    skills = load_skills()

    matched = []
    missing = []

    for skill in skills:

        if skill in job_description:

            if skill in resume_text:
                matched.append(skill.title())
            else:
                missing.append(skill.title())

    return matched, missing


def extract_email(text):

    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)

    if match:
        return match.group()

    return "Not Found"


def extract_phone(text):

    match = re.search(r'(\+91[\-\s]?)?[6-9]\d{9}', text)

    if match:
        return match.group()

    return "Not Found"


def extract_name(text):

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if len(line) > 3 and len(line.split()) <= 3:
            return line

    return "Unknown"