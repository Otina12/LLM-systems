import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def _norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\+\#\.\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@staticmethod
def extract_job_terms_tfidf(job_description, resume_text, top_k = 40, bonus_job_only = 0.25):
    job = _norm(job_description)
    resume = _norm(resume_text)

    vec = TfidfVectorizer(stop_words = 'english', ngram_range = (1, 3), min_df = 1, max_df = 1.0)

    X = vec.fit_transform([job, resume])
    terms = vec.get_feature_names_out()

    job_scores = X[0].toarray().ravel()
    resume_scores = X[1].toarray().ravel()

    in_job = job_scores > 0.0
    in_resume = resume_scores > 0.0

    job_only = in_job & (~in_resume)

    boosted = job_scores.copy()
    boosted[job_only] = boosted[job_only] + bonus_job_only

    idx = np.argsort(boosted)[::-1]
    ranked = [terms[i] for i in idx if in_job[i]]

    return ranked[:top_k]

@staticmethod
def match_terms(resume_text, job_terms):
    resume = f' {_norm(resume_text)} '
    present, missing = [], []

    for term in job_terms:
        term_norm = _norm(term)
        if not term_norm:
            continue

        if term_norm in resume:
            present.append(term)
        else:
            missing.append(term)

    return present, missing

@staticmethod
def build_context(present, missing, max_present = 15, max_missing = 20):
    present = present[:max_present]
    missing = missing[:max_missing]

    parts = []
    if present:
        parts.append('Keywords in both job and resume. You can use them freely: ' + ', '.join(present) + '.')
    if missing:
        parts.append('Keywords in job but not found in resume, add only if applicable: ' + ', '.join(missing) + '.')

    return '\n'.join(parts)