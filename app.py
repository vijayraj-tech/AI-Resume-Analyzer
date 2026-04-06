import streamlit as st
import PyPDF2
import re
import string
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ===============================================
# PAGE CONFIG
# ===============================================
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🚀", layout="wide")

# ===============================================
# UPDATED CSS - Only increased spacing in the two problematic sections
# ===============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, rgba(99,102,241,0.20), transparent 25%),
                radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 25%),
                linear-gradient(180deg, #07111f 0%, #0b1220 100%);
    color: #f8fafc;
}

.glass {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    backdrop-filter: blur(16px);
    margin-bottom: 2.2rem;
}

.hero-card {
    background: linear-gradient(135deg, rgba(79,70,229,0.35), rgba(168,85,247,0.28));
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 28px;
    padding: 32px 40px;
    text-align: center;
    margin-bottom: 32px;
}

.main-title { font-size: 2.8rem; font-weight: 800; text-align: center; margin-bottom: 8px; }
.sub-title { font-size: 1.1rem; color: #94a3b8; text-align: center; margin-bottom: 2rem; }

.metric-card {
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 22px;
    padding: 22px 18px;
    text-align: center;
}

.metric-label { color: #94a3b8; font-size: 0.95rem; font-weight: 600; margin-bottom: 8px; }
.metric-value { font-size: 2.4rem; font-weight: 800; color: white; }
.metric-sub { font-size: 0.9rem; color: #cbd5e1; }

/* === FIXED SPACING FOR TOP SECTION (metrics + charts) === */
.metrics-row { 
    display: grid; 
    grid-template-columns: repeat(4, 1fr); 
    gap: 24px; 
    margin-bottom: 32px; 
}
.secondary-metrics { 
    display: grid; 
    grid-template-columns: repeat(3, 1fr); 
    gap: 24px; 
    margin-bottom: 40px; 
}
.chart-row { 
    display: grid; 
    grid-template-columns: 1.15fr 0.85fr; 
    gap: 28px; 
    margin-bottom: 36px; 
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.2rem;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.red-banner {
    background: #b91c1c;
    color: white;
    padding: 16px 24px;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 36px;   /* extra space after red banner */
}

/* === FIXED SPACING FOR BOTTOM SECTION (table + reco + ATS) === */
.reco-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 28px;              /* more space between the three cards */
    margin-bottom: 36px;
}

.reco-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 20px;
    padding: 24px;
    height: 100%;
    text-align: center;
}

.low-alignment {
    background: rgba(239,68,68,0.15);
    color: #fda4af;
    border: 1px solid rgba(239,68,68,0.4);
    padding: 14px 32px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    margin: 4px 6px 4px 0;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.good-badge { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
.danger-badge { background: rgba(239,68,68,0.15); color: #fda4af; border: 1px solid rgba(239,68,68,0.3); }
</style>
""", unsafe_allow_html=True)

# ===============================================
# TITLE + HERO + INPUTS (unchanged)
# ===============================================
st.markdown("<div class='main-title'> AI Resume Analyzer </div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Beautiful ATS-style dashboard with match scoring, category insights, charts, keyword analysis, and smart suggestions.</div>", unsafe_allow_html=True)

st.markdown("""
<div class='hero-card'>
    <h1 style="font-size:2.4rem; margin-bottom:12px;">Build a recruiter-level resume intelligence dashboard</h1>
    <p style="font-size:1.05rem; color:#cbd5e1;">Upload a resume, paste a job description, and instantly see your match score, skill coverage, missing keywords, strong sections, weak sections, and visual analytics — all in one premium UI.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("<div class='glass'><div class='section-title'>📄 Upload Resume</div></div>", unsafe_allow_html=True)
    resume_file = st.file_uploader("Choose a PDF resume", type=["pdf"], label_visibility="collapsed")
    st.caption("Upload a clean PDF for best parsing result.")

with col2:
    st.markdown("<div class='glass'><div class='section-title'>📋 Paste Job Description</div></div>", unsafe_allow_html=True)
    job_desc = st.text_area("Paste the JD here", height=220, placeholder="Paste Software Engineer / SDE / Backend / Frontend job description here...", label_visibility="collapsed")
    st.caption("Paste a detailed JD to get better keyword and similarity analysis.")

# ===============================================
# YOUR ORIGINAL ANALYSIS LOGIC (unchanged)
# ===============================================
STOPWORDS = {"the","a","an","and","or","of","to","in","for","on","with","by","is","are","as","at","be","this","that","it","from","will","can","we","you","your","our","their","they","them","using","use","used","have","has","had","should","must","who","how","when","where","what","why","able","ability","role","job","work","working","years","year","good","strong","experience","skills","knowledge","team","teams","candidate","candidates","applications","application","software","engineer","engineering","developer","develop","maintain","build"}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def tokenize(text: str):
    return [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]

def category_score(text: str, category_dict: dict):
    scores = {}
    for category, keywords in category_dict.items():
        count = sum(1 for kw in keywords if kw.lower() in text)
        matched = [kw for kw in keywords if kw.lower() in text]
        scores[category] = {"score": count, "matched": matched, "total": len(keywords)}
    return scores

def normalize_category_df(cat_scores: dict):
    rows = []
    for cat, data in cat_scores.items():
        percent = round((data["score"] / data["total"]) * 100, 1) if data["total"] else 0
        rows.append({"Category": cat, "Matched Keywords": data["score"], "Total Keywords": data["total"], "Coverage %": percent})
    return pd.DataFrame(rows).sort_values("Coverage %", ascending=False)

CATEGORY_KEYWORDS = {
    "Programming": ["python", "java", "javascript", "typescript", "c++", "sql", "oop", "dsa"],
    "Frontend": ["react", "html", "css", "tailwind", "nextjs", "redux", "ui", "responsive"],
    "Backend": ["nodejs", "express", "spring boot", "django", "flask", "rest api", "microservices", "authentication"],
    "Database": ["mysql", "mongodb", "postgresql", "database", "sql", "orm", "query", "redis"],
    "Cloud & DevOps": ["aws", "gcp", "docker", "kubernetes", "ci/cd", "deployment", "github actions", "linux"],
    "Testing": ["unit testing", "integration testing", "debugging", "jest", "pytest", "test cases"],
    "CS Fundamentals": ["data structures", "algorithms", "os", "dbms", "system design", "computer networks"],
    "Soft Skills": ["communication", "leadership", "teamwork", "collaboration", "problem solving", "ownership"]
}

if resume_file and job_desc.strip():
    resume_raw = extract_pdf_text(resume_file)
    resume_text = clean_text(resume_raw)
    jd_text = clean_text(job_desc)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    match_score = round(similarity * 100, 2)

    resume_words = set(tokenize(resume_text))
    jd_words = set(tokenize(jd_text))
    matched_keywords = sorted(list(jd_words.intersection(resume_words)))
    missing_keywords = sorted(list(jd_words.difference(resume_words)))

    category_scores = category_score(resume_text, CATEGORY_KEYWORDS)
    category_df = normalize_category_df(category_scores)

    keyword_match_percent = round((len(matched_keywords) / max(len(jd_words), 1)) * 100, 2)
    ats_readiness = round((match_score * 0.55) + (keyword_match_percent * 0.45), 2)

    impact_score = 84 if len(matched_keywords) > 25 else 68 if len(matched_keywords) > 12 else 49
    brevity_score = 62 if len(resume_raw.split()) > 450 else 80 if len(resume_raw.split()) > 250 else 55
    style_score = 76 if any(word in resume_text for word in ["developed", "built", "implemented", "designed", "optimized"]) else 58

    # ===============================================
    # TOP SECTION - Now perfectly spaced
    # ===============================================
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Overall Match</div><div class='metric-value'>{match_score}%</div><div class='metric-sub'>Resume ↔ Job Description similarity</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='metric-card'><div class='metric-label'>ATS Readiness</div><div class='metric-value'>{ats_readiness}%</div><div class='metric-sub'>Combined similarity + keyword coverage</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Keyword Coverage</div><div class='metric-value'>{keyword_match_percent}%</div><div class='metric-sub'>How many JD keywords appear in resume</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Matched Keywords</div><div class='metric-value'>{len(matched_keywords)}</div><div class='metric-sub'>Detected relevant keywords</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='red-banner'>❌ Low match. Resume needs significant improvement for this role.</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Impact</div><div class='metric-value'>{impact_score}/100</div><div class='metric-sub'>Based on matched technical relevance</div></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Brevity</div><div class='metric-value'>{brevity_score}/100</div><div class='metric-sub'>Estimated from resume content length</div></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Style</div><div class='metric-value'>{style_score}/100</div><div class='metric-sub'>Action-oriented writing quality signal</div></div>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='glass'><div class='section-title'>📊 Category Coverage Analysis</div></div>", unsafe_allow_html=True)
        bar_fig = px.bar(category_df, x="Coverage %", y="Category", orientation="h", text="Coverage %", color="Coverage %", color_continuous_scale="Blues")
        bar_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
        st.plotly_chart(bar_fig, use_container_width=True)

    with c2:
        st.markdown("<div class='glass'><div class='section-title'>🧠 Score Distribution</div></div>", unsafe_allow_html=True)
        donut_fig = px.pie(pd.DataFrame({"Metric": ["Match Score", "Remaining Gap"], "Value": [match_score, 100-match_score]}), names="Metric", values="Value", hole=0.70, color_discrete_map={"Match Score":"#6366f1", "Remaining Gap":"#1e293b"})
        donut_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
        st.plotly_chart(donut_fig, use_container_width=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("<div class='glass'><div class='section-title'>🔑 Top JD Keywords</div></div>", unsafe_allow_html=True)
        top_jd = pd.DataFrame(Counter(tokenize(jd_text)).most_common(15), columns=["Keyword", "Frequency"])
        if not top_jd.empty:
            key_fig = px.bar(top_jd.sort_values("Frequency"), x="Frequency", y="Keyword", orientation="h", color="Frequency", color_continuous_scale="Purples")
            key_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420)
            st.plotly_chart(key_fig, use_container_width=True)

    with c4:
        st.markdown("<div class='glass'><div class='section-title'>📈 Resume vs JD Radar</div></div>", unsafe_allow_html=True)
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=[100]*len(category_df), theta=category_df["Category"], fill='toself', name='Ideal JD', line_color='#334155'))
        radar_fig.add_trace(go.Scatterpolar(r=category_df["Coverage %"].tolist(), theta=category_df["Category"], fill='toself', name='Your Resume', line_color='#8b5cf6'))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=420, showlegend=True)
        st.plotly_chart(radar_fig, use_container_width=True)

    # ===============================================
    # KEYWORDS + TABLE + RECO + ATS (bottom section now perfectly spaced)
    # ===============================================
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("<div class='glass'><div class='section-title'>✅ Matched Keywords</div></div>", unsafe_allow_html=True)
        st.markdown("".join([f"<span class='badge good-badge'>{kw}</span>" for kw in matched_keywords[:35]]), unsafe_allow_html=True)

    with p2:
        st.markdown("<div class='glass'><div class='section-title'>❌ Missing Keywords</div></div>", unsafe_allow_html=True)
        st.markdown("".join([f"<span class='badge danger-badge'>{kw}</span>" for kw in missing_keywords[:40]]), unsafe_allow_html=True)

    st.markdown("<div class='glass'><div class='section-title'>📋 Detailed Category Table</div></div>", unsafe_allow_html=True)
    st.dataframe(category_df, use_container_width=True, hide_index=True)

    # Recommendation cards (more space)
    strongest = category_df.iloc[0]["Category"] if not category_df.empty else "Database"
    weakest = category_df.iloc[-1]["Category"] if not category_df.empty else "Soft Skills"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class='reco-card'>
            <div class='section-title'>🚀 Resume Optimization</div>
            <p style="color:#cbd5e1; line-height:1.6;">Add more project bullets that directly mirror the job description. Use words like built, implemented, optimized, deployed, integrated.</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class='reco-card'>
            <div class='section-title'>📌 Strongest vs Weakest Area</div>
            <p style="color:#cbd5e1; line-height:1.6;"><strong>Strongest area:</strong> {strongest}<br><br><strong>Weakest area:</strong> {weakest}<br><br>Improve the weak category by adding project work, tech stack terms, and real impact bullets.</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class='reco-card'>
            <div class='section-title'>🎯 Final Verdict</div>
            <p style="color:#cbd5e1; line-height:1.6;">You are close, but not recruiter-ready for this JD yet.</p>
        </div>
        """, unsafe_allow_html=True)

    # ATS Review Summary
    st.markdown("<div class='glass'><div class='section-title'>🧾 ATS Review Summary</div></div>", unsafe_allow_html=True)
    colx, coly = st.columns([1.3, 1])
    with colx:
        st.write(f"**Overall Match Score:** {match_score}%")
        st.write(f"**ATS Readiness:** {ats_readiness}%")
        st.write(f"**Keyword Coverage:** {keyword_match_percent}%")
        st.write(f"**Matched Keywords Found:** {len(matched_keywords)}")
        st.write(f"**Missing Keywords Found:** {len(missing_keywords)}")
    with coly:
        st.markdown("<div class='low-alignment'>Low ATS Alignment</div>", unsafe_allow_html=True)

else:
    st.info("👆 Upload your resume PDF and paste a job description to get the full recruiter-level analysis.")