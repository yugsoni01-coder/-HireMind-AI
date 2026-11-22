
import streamlit as st
import json
import re
import time
import hashlib
import sqlite3
import os
import pandas as pd
import plotly.express as px
import pymupdf
import pytesseract

from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HireMind AI",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

# Change this whenever the AI prompt or analysis logic changes.
PROMPT_VERSION = "hiremind_v3"

MAX_RESUME_CHARS = 9000
MAX_JD_CHARS = 5000

# Delay between new Groq requests.
# Cached resumes do not need this delay.
REQUEST_DELAY_SECONDS = 35

# Keep database beside this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DB = os.path.join(BASE_DIR, "hiremind_cache.db")

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=5000
)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==========================================
# 2. SQLITE PERSISTENT CACHE
# ==========================================

def initialize_cache():

    connection = sqlite3.connect(CACHE_DB)

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS resume_cache (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                ocr_pages INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()

    finally:
        connection.close()


def make_cache_key(pdf_bytes, job_description):

    # Hash actual PDF content, not filename.
    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # Normalize JD whitespace and capitalization.
    normalized_jd = " ".join(
        job_description.strip().lower().split()
    )

    jd_hash = hashlib.sha256(
        normalized_jd.encode("utf-8")
    ).hexdigest()

    unique_data = (
        f"{pdf_hash}|{jd_hash}|"
        f"{MODEL_NAME}|{PROMPT_VERSION}"
    )

    return hashlib.sha256(
        unique_data.encode("utf-8")
    ).hexdigest()


def get_cached_result(cache_key):

    connection = sqlite3.connect(CACHE_DB)

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT result_json, ocr_pages
            FROM resume_cache
            WHERE cache_key = ?
            """,
            (cache_key,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        result = json.loads(row[0])

        if not isinstance(result, dict):
            return None

        return {
            "result": result,
            "ocr_pages": int(row[1] or 0)
        }

    finally:
        connection.close()


def save_cached_result(
    cache_key,
    result,
    ocr_pages
):

    if not isinstance(result, dict):
        raise ValueError("Cannot cache an invalid candidate result.")

    ocr_pages = max(0, int(ocr_pages or 0))

    connection = sqlite3.connect(CACHE_DB)

    try:

        result_json = json.dumps(
            result,
            ensure_ascii=False
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO resume_cache
            (cache_key, result_json, ocr_pages)
            VALUES (?, ?, ?)
            """,
            (
                cache_key,
                result_json,
                ocr_pages
            )
        )

        connection.commit()

    finally:
        connection.close()


def get_cache_count():

    connection = sqlite3.connect(CACHE_DB)

    try:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM resume_cache"
        )

        return cursor.fetchone()[0]

    finally:
        connection.close()


def clear_persistent_cache():

    connection = sqlite3.connect(CACHE_DB)

    try:

        connection.execute(
            "DELETE FROM resume_cache"
        )

        connection.commit()

    finally:
        connection.close()


# Initialize database when application starts.
initialize_cache()


# ==========================================
# 3. PDF EXTRACTION AND OCR
# ==========================================

def extract_resume_text(pdf_bytes):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    if document.needs_pass:

        document.close()

        raise ValueError(
            "This PDF is password protected."
        )

    extracted_pages = []
    ocr_pages = 0

    try:

        if len(document) == 0:

            raise ValueError(
                "This PDF contains no pages."
            )

        for page_number, page in enumerate(document):

            page_text = page.get_text("text").strip()

            # Use OCR only when normal extraction is insufficient.
            if len(page_text) < 40:

                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    colorspace=pymupdf.csRGB,
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )

                ocr_text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                ).strip()

                ocr_pages += 1

                if len(ocr_text) > len(page_text):
                    page_text = ocr_text

            extracted_pages.append(
                f"\n--- Page {page_number + 1} ---\n"
                f"{page_text}"
            )

    finally:
        document.close()

    resume_text = "\n".join(
        extracted_pages
    ).strip()

    if not resume_text:

        raise ValueError(
            "No readable text found, even after OCR."
        )

    return resume_text, ocr_pages


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_response_text(response):

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])

        return "\n".join(parts)

    return str(content)

def parse_json_response(response_text):
    
    if not response_text:
        raise ValueError("AI returned an empty response.")

    response_text = str(response_text).strip()

    response_text = re.sub(
        r"```(?:json)?",
        "",
        response_text,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    start = response_text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in AI response."
        )

    decoder = json.JSONDecoder()

    try:
        result, end_position = decoder.raw_decode(
            response_text[start:]
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"AI returned incomplete or invalid JSON: "
            f"{error}. "
            "The response may have been truncated. "
            "Please retry."
        )

    if not isinstance(result, dict):
        raise ValueError("AI response is not a JSON object.")

    return result


def normalize_skill(skill):

    return str(skill).strip().casefold()


def clean_string_list(values):

    if not isinstance(values, list):
        return []

    cleaned = []

    for item in values:

        if isinstance(item, str) and item.strip():

            value = item.strip()

            if value not in cleaned:
                cleaned.append(value)

    return cleaned


def prepare_skill_data(data):

    if not isinstance(data, dict):

        raise ValueError(
            "Invalid skill matching data."
        )

    required = clean_string_list(
        data.get("required_skills", [])
    )

    matched = clean_string_list(
        data.get("matched_skills", [])
    )

    # Keep only skills required by the JD.
    required_map = {
        normalize_skill(skill): skill
        for skill in required
    }

    required_keys = set(required_map)

    matched_map = {}

    for skill in matched:

        key = normalize_skill(skill)

        if key in required_keys:
            matched_map[key] = required_map[key]

    # Every required skill not matched becomes missing.
    missing_map = {}

    for key, skill in required_map.items():

        if key not in matched_map:
            missing_map[key] = skill

    data["required_skills"] = list(
        required_map.values()
    )

    data["matched_skills"] = list(
        matched_map.values()
    )

    data["missing_skills"] = list(
        missing_map.values()
    )

    # Keep evidence only for matched skills.
    evidence = data.get("skill_evidence", [])

    if not isinstance(evidence, list):
        evidence = []

    matched_keys = set(matched_map)

    cleaned_evidence = []
    evidence_keys = set()

    for item in evidence:

        if not isinstance(item, dict):
            continue

        skill = item.get("skill")
        evidence_text = item.get("evidence")

        if not isinstance(skill, str):
            continue

        key = normalize_skill(skill)

        if key not in matched_keys:
            continue

        if key in evidence_keys:
            continue

        if not isinstance(evidence_text, str):
            continue

        if not evidence_text.strip():
            continue

        cleaned_evidence.append({

            "skill": required_map[key],

            "evidence": evidence_text.strip()

        })

        evidence_keys.add(key)

    data["skill_evidence"] = cleaned_evidence

    # Remove unexpected fields returned by the model.
    allowed_fields = {

        "candidate_name",
        "screening_result",
        "required_skills",
        "matched_skills",
        "missing_skills",
        "skill_evidence",
        "candidate_education",
        "education_meets_requirement",
        "required_experience_requirement",
        "candidate_experience_years",
        "candidate_experience_evidence",
        "experience_meets_requirement",
        "strengths",
        "areas_requiring_verification",
        "suggested_interview_questions"

    }

    return {
        key: value
        for key, value in data.items()
        if key in allowed_fields
    }


def format_experience_requirement(value):

    if value is None:
        return "Not specified"

    value = str(value).strip()

    if not value:
        return "Not specified"

    if re.search(
        r"\byears?\b",
        value,
        re.IGNORECASE
    ):

        return value

    return f"{value} years"


def get_numeric_experience(value):

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):

        if value < 0:
            return None

        return float(value)

    return None


def build_candidate_evaluation(data):

    """
    Build evaluation locally from structured AI data.
    This prevents contradictory independent AI reports.
    """

    strengths = clean_string_list(
        data.get("strengths", [])
    )

    questions = clean_string_list(
        data.get("suggested_interview_questions", [])
    )

    matched = data.get("matched_skills", [])
    missing = data.get("missing_skills", [])

    experience_evidence = (
        data.get("candidate_experience_evidence")
        or "No professional employment evidence was provided."
    )

    experience_years = data.get(
        "candidate_experience_years"
    )

    experience_requirement = format_experience_requirement(
        data.get("required_experience_requirement")
    )

    lines = []

    lines.append("### Candidate Strengths")

    if strengths:

        for item in strengths:
            lines.append(f"- {item}")

    else:
        lines.append("- No additional strengths were identified.")

    lines.append("")
    lines.append("### Matched Skills")

    if matched:

        for skill in matched:
            lines.append(f"- {skill}")

    else:
        lines.append(
            "- No required skills were confirmed as matched."
        )

    lines.append("")
    lines.append("### Missing Skills")

    if missing:

        for skill in missing:
            lines.append(f"- {skill}")

    else:
        lines.append(
            "- No missing required skills identified."
        )

    lines.append("")
    lines.append("### Professional Experience")

    lines.append(str(experience_evidence))

    if experience_years is not None:

        lines.append(
            f"Verified experience duration: "
            f"{experience_years:g} years."
        )

    else:

        lines.append(
            "Verified experience duration: Not established "
            "from the available resume information."
        )

    lines.append("")
    lines.append("### Experience Requirement")

    lines.append(
        f"Required experience: {experience_requirement}"
    )

    experience_status = data.get(
        "experience_meets_requirement"
    )

    if experience_status is True:

        lines.append(
            "Experience eligibility: Requirement met."
        )

    elif experience_status is False:

        lines.append(
            "Experience eligibility: Requirement not met."
        )

    else:

        lines.append(
            "Experience eligibility: Requires verification."
        )

    lines.append("")
    lines.append("### Areas Requiring Verification")

    verification = clean_string_list(
        data.get("areas_requiring_verification", [])
    )

    if experience_years is None:

        verification.append(
            "Verify the duration of professional employment."
        )

    if missing:

        verification.append(
            "Assess the candidate's knowledge of the missing required skills."
        )

    verification = list(dict.fromkeys(verification))

    if verification:

        for item in verification:
            lines.append(f"- {item}")

    else:

        lines.append(
            "- No additional verification points identified."
        )

    lines.append("")
    lines.append("### Suggested Technical Interview Questions")

    if questions:

        for index, question in enumerate(
            questions,
            start=1
        ):

            lines.append(f"{index}. {question}")

    else:

        lines.append(
            "Interview questions were not generated."
        )

    lines.append("")
    lines.append("### Overall Assessment")

    if missing and experience_years is None:

        lines.append(
            "The candidate demonstrates evidence of relevant "
            "technical skills. However, the missing required "
            "skills and professional experience duration should "
            "be verified before making a recruitment decision."
        )

    elif missing:

        lines.append(
            "The candidate demonstrates relevant skills, but "
            "some job requirements are not evidenced in the resume. "
            "Further technical assessment is recommended."
        )

    elif experience_years is None:

        lines.append(
            "The candidate demonstrates the required technical "
            "skills. Professional experience duration requires "
            "verification before making a final decision."
        )

    else:

        lines.append(
            "The candidate demonstrates evidence of the required "
            "technical skills and documented experience. "
            "Further interview assessment is recommended."
        )

    lines.append("")
    lines.append(
        "This assessment is decision support, not an automatic "
        "hiring recommendation."
    )

    return "\n".join(lines)

# ==========================================
# 4A. CANDIDATE CONTACT INTELLIGENCE
# ==========================================

def extract_contact_information(resume_text):
    """
    Extract contact details locally using Python regex.
    No Groq API calls or tokens are used.
    """

    if not isinstance(resume_text, str):
        resume_text = ""

    # Email addresses
    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        resume_text,
        flags=re.IGNORECASE
    )
    emails = list(dict.fromkeys(email.lower() for email in emails))

    # Phone numbers (keeps common international and Indian formats)
    phone_candidates = re.findall(
        r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{5,10}(?!\w)",
        resume_text
    )
    phone_numbers = []
    for phone in phone_candidates:
        cleaned_phone = re.sub(r"[^+\d]", "", phone)
        digits_only = re.sub(r"\D", "", cleaned_phone)
        if 10 <= len(digits_only) <= 13 and cleaned_phone not in phone_numbers:
            phone_numbers.append(cleaned_phone)

    # Professional profile URLs
    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\]\[()]+",
        resume_text,
        flags=re.IGNORECASE
    )
    urls = [url.rstrip(".,;:") for url in urls]

    linkedin_urls = list(dict.fromkeys(
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
        for url in urls if "linkedin.com" in url.lower()
    ))

    github_urls = list(dict.fromkeys(
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
        for url in urls if "github.com" in url.lower()
    ))

    portfolio_urls = list(dict.fromkeys(
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
        for url in urls
        if "linkedin.com" not in url.lower()
        and "github.com" not in url.lower()
    ))

    # Only collect explicitly labelled location/address lines.
    locations = []
    location_matches = re.findall(
        r"(?im)^\s*(?:current\s+)?(?:location|address|city|residence)\s*[:\-]\s*(.+?)\s*$",
        resume_text
    )
    for location in location_matches:
        cleaned_location = re.sub(r"\s+", " ", location).strip(" ,;|")
        if cleaned_location and cleaned_location not in locations:
            locations.append(cleaned_location)

    return {
        "emails": emails,
        "phone_numbers": phone_numbers,
        "locations": locations,
        "linkedin_urls": linkedin_urls,
        "github_urls": github_urls,
        "portfolio_urls": portfolio_urls
    }


# ==========================================
# 5. UNIFIED AI RECRUITMENT AGENT
# ==========================================

def unified_recruitment_agent(
    resume_text,
    job_description
):

    resume_text = resume_text[:MAX_RESUME_CHARS]
    job_description = job_description[:MAX_JD_CHARS]

    prompt = f"""
You are HireMind AI, a professional recruitment analysis system.

Perform these tasks together:

1. Resume screening
2. Technical skill matching
3. Candidate information extraction
4. Interview question generation

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ONLY one valid JSON object.

Use this exact structure:

{{
    "candidate_name": null,
    "screening_result": "Resume screening report",

    "required_skills": [],
    "matched_skills": [],
    "missing_skills": [],

    "skill_evidence": [
        {{
            "skill": "Python",
            "evidence": "Exact supporting evidence from resume"
        }}
    ],

    "candidate_education": null,
    "education_meets_requirement": null,

    "required_experience_requirement": null,
    "candidate_experience_years": null,
    "candidate_experience_evidence": null,
    "experience_meets_requirement": null,

    "strengths": [],
    "areas_requiring_verification": [],
    "suggested_interview_questions": []
}}

RULES:

RESUME SCREENING:
- Extract candidate name, education, employment, projects,
  technical skills and certifications.
- Distinguish professional employment from academic projects.
- Never invent information.
- Preserve important dates and job titles.
- Mention unavailable information clearly.

SKILL MATCHING:
- Identify technical skills explicitly required by the JD.
- Match only skills explicitly supported by resume evidence.
- Do not assume related skills are identical.
- For every matched skill, provide a short, specific evidence
  excerpt or a precise resume section describing its use.
- Never use the entire skills list or entire resume as evidence.
- If the resume only lists a skill without describing its use,
  it may still be counted as listed evidence, but say that it
  is listed rather than demonstrated.
- Every matched or missing skill must appear in required_skills.
- Do not mark a skill matched without identifiable resume evidence.
- Do not invent skills or evidence.

EXPERIENCE:
- Extract professional employment evidence.
- Include employer, job title and dates when available.
- Do not count academic projects as professional employment.
- Do not assume experience duration.
- candidate_experience_years must be a number only when
  duration can be reliably established.
- Otherwise use null.
- required_experience_requirement should contain the JD wording,
  such as "0-1 year" or "2 years".
- Do not add the word years twice.
- experience_meets_requirement must be true, false or null.
- Use null when duration cannot be verified.

EDUCATION:
- Compare candidate education with the JD.
- Use true, false or null.
- Use null when evidence is insufficient.

STRENGTHS:
- Include only strengths supported by the resume.
- Do not claim a skill is missing if it is matched.
- Do not claim a skill is matched if it is missing.

VERIFICATION:
- Identify missing skills and uncertain experience.
- Do not treat unknown information as negative evidence.

INTERVIEW QUESTIONS:
- Generate relevant technical interview questions.
- Questions should relate to the JD and candidate profile.

OUTPUT:
- Valid JSON only.
- Use double quotes.
- No Markdown.
- No explanation outside JSON.
- Use null for unknown values.
- Use empty arrays when information is unavailable.
- Never invent candidate details.
"""

    # Exactly one Groq API call.
    response = llm.invoke(prompt)

    response_text = get_response_text(response)

    ai_data = parse_json_response(response_text)

    # Validate and clean the AI response.
    ai_data = prepare_skill_data(ai_data)

    # Normalize candidate name and education.
    if not isinstance(ai_data.get("candidate_name"), str):
        ai_data["candidate_name"] = None

    if not isinstance(ai_data.get("candidate_education"), str):
        ai_data["candidate_education"] = None

    if not isinstance(ai_data.get("screening_result"), str):
        ai_data["screening_result"] = (
            "No screening report available."
        )

    # Normalize experience duration.
    ai_data["candidate_experience_years"] = (
        get_numeric_experience(
            ai_data.get("candidate_experience_years")
        )
    )

    # Validate experience eligibility.
    if ai_data["candidate_experience_years"] is None:

        ai_data["experience_meets_requirement"] = None

    elif ai_data.get("experience_meets_requirement") not in (
        True, False
    ):

        ai_data["experience_meets_requirement"] = None

    # Validate education eligibility.
    if ai_data.get("education_meets_requirement") not in (
        True, False
    ):

        ai_data["education_meets_requirement"] = None

    # Normalize evidence.
    if not isinstance(
        ai_data.get("candidate_experience_evidence"),
        str
    ):

        ai_data["candidate_experience_evidence"] = (
            "No clear professional employment details were extracted."
        )

    # Ensure arrays have correct data types.
    ai_data["strengths"] = clean_string_list(
        ai_data.get("strengths", [])
    )

    ai_data["areas_requiring_verification"] = clean_string_list(
        ai_data.get("areas_requiring_verification", [])
    )

    ai_data["suggested_interview_questions"] = clean_string_list(
        ai_data.get("suggested_interview_questions", [])
    )

    # Build evaluation locally.
    evaluation_result = build_candidate_evaluation(
        ai_data
    )

    result = {

        "screening_result": ai_data["screening_result"],

        "skill_result": json.dumps(
            ai_data,
            indent=4,
            ensure_ascii=False
        ),

        "skill_data": ai_data,

        "evaluation_result": evaluation_result

    }

    return result


# ==========================================
# 6. SESSION STATE
# ==========================================

if "recruitment_results" not in st.session_state:
    st.session_state["recruitment_results"] = None

if "candidate_statuses" not in st.session_state:
    st.session_state["candidate_statuses"] = {}

if "cache_hits" not in st.session_state:
    st.session_state["cache_hits"] = 0

if "cache_misses" not in st.session_state:
    st.session_state["cache_misses"] = 0

if "groq_api_calls" not in st.session_state:
    st.session_state["groq_api_calls"] = 0


# ==========================================
# 7. DASHBOARD HEADER
# ==========================================

st.title("🤖 HireMind AI")

st.subheader(
    "AI-Powered Intelligent Recruitment Screening System"
)

st.caption(
    "Intelligent resume screening | Skill matching | "
    "Candidate evaluation | Persistent AI caching"
)

st.divider()


# ==========================================
# 8. WORKFLOW VISUALIZATION
# ==========================================

with st.expander("View AI Recruitment Workflow"):

    st.graphviz_chart("""
    digraph {
        rankdir=LR;

        node [shape=box, style="rounded,filled",
              fillcolor="#e0f2fe"];

        A [label="PDF Extraction\\nPyMuPDF + OCR"];
        B [label="Persistent Cache\\nSQLite"];
        C [label="Unified AI Analysis\\nOne Groq API Call"];
        D [label="Local Evaluation\\nPython"];
        E [label="Candidate Dashboard"];

        A -> B;
        B -> C [label="Cache Miss"];
        B -> D [label="Cache Hit"];
        C -> D;
        D -> E;
    }
    """)


# ==========================================
# 9. INPUT SECTION
# ==========================================

st.header("Candidate Analysis")

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload Multiple Candidate Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

run_workflow = st.button(
    "🚀 Start AI Recruitment Workflow",
    width="stretch"
)


# ==========================================
# 10. MULTIPLE RESUME SCREENING
# ==========================================

if run_workflow:

    if not job_description.strip() or not uploaded_files:

        st.error(
            "Please enter a job description and upload resumes."
        )

    else:

        all_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)

        # Tracks the time of the last new API request.
        last_api_call_time = None

        for index, uploaded_file in enumerate(uploaded_files):

            status_text.write(
                f"Processing candidate {index + 1} of "
                f"{total_files}: {uploaded_file.name}"
            )

            try:

                pdf_bytes = uploaded_file.getvalue()

                # ----------------------------------
                # CREATE CACHE KEY
                # ----------------------------------

                cache_key = make_cache_key(
                    pdf_bytes,
                    job_description
                )

                # ----------------------------------
                # CHECK SQLITE CACHE
                # ----------------------------------

                cached_item = get_cached_result(
                    cache_key
                )

                if cached_item is not None:
                    
                    result = cached_item["result"]
                    ocr_pages = cached_item["ocr_pages"]

                    processing_method = "Persistent Cache"

                    st.session_state["cache_hits"] += 1

                    # Extract contact details locally, including for
                    # older cached resumes that have no contact_data.
                    resume_text, contact_ocr_pages = extract_resume_text(
                        pdf_bytes
                    )

                    # Use the latest OCR count for display.
                    ocr_pages = contact_ocr_pages

                else:

                    st.session_state["cache_misses"] += 1

                    # ----------------------------------
                    # LOCAL PDF EXTRACTION AND OCR
                    # ----------------------------------

                    resume_text, ocr_pages = extract_resume_text(
                        pdf_bytes
                    )

                    if len(resume_text.strip()) < 30:

                        raise ValueError(
                            "Insufficient readable resume text."
                        )

                    # ----------------------------------
                    # WAIT BEFORE NEXT NEW API REQUEST
                    # ----------------------------------

                    if last_api_call_time is not None:

                        elapsed = (
                            time.time() - last_api_call_time
                        )

                        remaining_delay = (
                            REQUEST_DELAY_SECONDS - elapsed
                        )

                        if remaining_delay > 0:

                            status_text.info(
                                "Waiting before the next new "
                                "Groq API request..."
                            )

                            time.sleep(remaining_delay)

                    # ----------------------------------
                    # ONE GROQ API CALL
                    # ----------------------------------

                    status_text.write(
                        f"Running AI analysis for "
                        f"{uploaded_file.name}..."
                    )

                    result = unified_recruitment_agent(
                        resume_text,
                        job_description
                    )

                    last_api_call_time = time.time()
                    
                    
                    # Extract contact information locally.
                    # This does not use Groq tokens.

                    st.session_state["groq_api_calls"] += 1

                    # ----------------------------------
                    # SAVE SUCCESSFUL RESULT
                    # ----------------------------------

                    save_cached_result(
                        cache_key,
                        result,
                        ocr_pages
                    )

                    processing_method = "New AI Analysis"
                # ----------------------------------
                # CANDIDATE CONTACT INTELLIGENCE
                # ----------------------------------

                contact_data = extract_contact_information(
                    resume_text
                )

                # Store contact information with the result.
                # This also updates older cached results.
                result["contact_data"] = contact_data

                # Save the updated result to SQLite.
                save_cached_result(
                    cache_key,
                    result,
                    ocr_pages
                )
                # ----------------------------------
                # PREPARE CANDIDATE DATA
                # ----------------------------------

                data = result["skill_data"]

                required = data.get(
                    "required_skills", []
                )

                matched = data.get(
                    "matched_skills", []
                )

                missing = data.get(
                    "missing_skills", []
                )

                total_skills = len(required)
                matched_count = len(matched)
                missing_count = len(missing)

                match_percentage = (
                    matched_count / total_skills * 100
                    if total_skills > 0 else 0
                )

                all_results.append({

                    "file_name": uploaded_file.name,

                    "candidate_name": (
                        data.get("candidate_name")
                        or "Not detected"
                    ),

                    "required_skills": total_skills,

                    "matched_skills": matched_count,

                    "missing_skills": missing_count,

                    "match_percentage": round(
                        match_percentage, 2
                    ),

                    "ocr_pages": ocr_pages,

                    "processing_method": processing_method,

                    "status": "Completed",

                    "result": result

                })

            except Exception as error:

                error_message = str(error)

                if (
                    "429" in error_message
                    or "rate_limit" in error_message.lower()
                ):

                    error_message = (
                        "Groq API rate limit reached. "
                        "This resume was not cached. "
                        "Please wait for the quota reset and retry. "
                        f"Original error: {error}"
                    )

                all_results.append({

                    "file_name": uploaded_file.name,

                    "status": "Failed",

                    "error": error_message

                })

            progress_bar.progress(
                (index + 1) / total_files
            )

        st.session_state["recruitment_results"] = all_results

        status_text.success(
            "Candidate screening process finished!"
        )

        completed_count = sum(
            1 for item in all_results
            if item["status"] == "Completed"
        )

        failed_count = total_files - completed_count

        st.success(
            f"Successfully screened: {completed_count} | "
            f"Failed: {failed_count}"
        )


# ==========================================
# 11. CANDIDATE DASHBOARD
# ==========================================

if st.session_state["recruitment_results"] is not None:

    results = st.session_state["recruitment_results"]

    successful_results = [
        candidate for candidate in results
        if candidate["status"] == "Completed"
    ]

    failed_results = [
        candidate for candidate in results
        if candidate["status"] == "Failed"
    ]

    st.divider()

    st.header("🏆 Candidate Ranking Dashboard")

    # ======================================
    # CACHE STATISTICS
    # ======================================

    st.subheader("⚡ AI Token-Saving Statistics")

    cache_col1, cache_col2, cache_col3, cache_col4 = st.columns(4)

    cache_col1.metric(
        "Cache Hits",
        st.session_state["cache_hits"]
    )

    cache_col2.metric(
        "Cache Misses",
        st.session_state["cache_misses"]
    )

    cache_col3.metric(
        "Groq API Calls",
        st.session_state["groq_api_calls"]
    )

    cache_col4.metric(
        "Saved API Calls",
        st.session_state["cache_hits"]
    )

    st.caption(
        f"Persistent cached results: {get_cache_count()}"
    )

    cache_col5, cache_col6 = st.columns([3, 1])

    with cache_col5:

        st.caption(
            "Cache is stored locally in hiremind_cache.db. "
            "Clearing it means previously processed resumes "
            "will require fresh AI analysis."
        )

    with cache_col6:

        if st.button(
            "🗑️ Clear Cache",
            width="stretch"
        ):

            clear_persistent_cache()

            st.session_state["cache_hits"] = 0
            st.session_state["cache_misses"] = 0
            st.session_state["groq_api_calls"] = 0

            st.success(
                "Persistent cache cleared."
            )

            st.rerun()

    st.divider()

    if successful_results:

        ranked_candidates = sorted(
            successful_results,
            key=lambda x: x["match_percentage"],
            reverse=True
        )
        
        # ==================================
        # ADVANCED CANDIDATE SEARCH & FILTERS
        # ==================================

        st.divider()

        st.header("🔎 Advanced Candidate Search & Filters")

        st.caption(
            "Search and filter screened candidates using "
            "their contact information, recruiter status, "
            "location and skill-match percentage."
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            candidate_search = st.text_input(
                "Search by name, email, phone or resume filename",
                placeholder="Enter candidate name, email or phone...",
                key="advanced_candidate_search"
            )

            selected_statuses = st.multiselect(
                "Filter by Recruiter Status",
                options=[
                    "Under Review",
                    "Shortlisted",
                    "Rejected"
                ],
                default=[
                    "Under Review",
                    "Shortlisted",
                    "Rejected"
                ],
                key="advanced_status_filter"
            )

        with filter_col2:

            minimum_match = st.slider(
                "Minimum Skill Match (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="advanced_match_filter"
            )

            location_search = st.text_input(
                "Filter by Candidate Location",
                placeholder="Example: Vadodara, Ahmedabad...",
                key="advanced_location_filter"
            )

        # ----------------------------------
        # APPLY FILTERS
        # ----------------------------------

        filtered_candidates = []

        for candidate in ranked_candidates:

            candidate_name = candidate.get(
                "candidate_name",
                "Not detected"
            )

            file_name = candidate.get(
                "file_name",
                ""
            )

            candidate_result = candidate.get("result") or {}

            if not isinstance(candidate_result, dict):
                candidate_result = {}

            contact_data = candidate_result.get("contact_data") or {}

            if not isinstance(contact_data, dict):
                contact_data = {}


            # Current recruiter status
            recruiter_status = st.session_state[
                "candidate_statuses"
            ].get(
                file_name,
                "Under Review"
            )

            # Contact information
            searchable_contact = " ".join([

               " ".join(contact_data.get("emails") or []),

              " ".join(contact_data.get("phone_numbers") or []),

              " ".join(contact_data.get("locations") or []),

               " ".join(contact_data.get("linkedin_urls") or []),

              " ".join(contact_data.get("github_urls") or []),

              " ".join(contact_data.get("portfolio_urls") or [])

            ])

            searchable_text = (
                f"{candidate_name} "
                f"{file_name} "
                f"{searchable_contact}"
            ).casefold()

            # Search filter
            if (
                candidate_search.strip()
                and candidate_search.strip().casefold()
                not in searchable_text
            ):
                continue

            # Recruiter status filter
            if recruiter_status not in selected_statuses:
                continue

            # Minimum skill match filter
            if candidate["match_percentage"] < minimum_match:
                continue

            # Location filter
            candidate_locations = " ".join(
                contact_data.get("locations") or []
            ).casefold()

            if (
                location_search.strip()
                and location_search.strip().casefold()
                not in candidate_locations
            ):
                continue

            filtered_candidates.append(candidate)
        # ----------------------------------
        # FILTER DIAGNOSTICS
        # ----------------------------------

        if not filtered_candidates:

            st.warning("Your resume was excluded by one or more filters.")

            for candidate in ranked_candidates:

                file_name = candidate.get("file_name", "")

                actual_status = st.session_state[
                    "candidate_statuses"
                ].get(file_name, "Under Review")

                actual_match = candidate.get("match_percentage", 0)

                candidate_result = candidate.get("result") or {}

                if not isinstance(candidate_result, dict):
                    candidate_result = {}

                contact_data = candidate_result.get("contact_data") or {}

                if not isinstance(contact_data, dict):
                    contact_data = {}

                actual_location = ", ".join(
                    contact_data.get("locations") or []
                )

                st.write(f"**Resume:** {file_name}")
                st.write(f"**Actual Skill Match:** {actual_match}%")
                st.write(f"**Recruiter Status:** {actual_status}")
                st.write(f"**Detected Location:** {actual_location or 'Not detected'}")

                if actual_status not in selected_statuses:
                    st.error("Excluded: Recruiter status does not match.")

                if actual_match < minimum_match:
                    st.error("Excluded: Minimum skill match is too high.")

                if (
                    location_search.strip()
                    and location_search.strip().casefold()
                    not in actual_location.casefold()
                ):
                    st.error("Excluded: Location does not match.")
        # ----------------------------------
        # FILTER RESULTS SUMMARY
        # ----------------------------------

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        filter_col1.metric(
            "Total Screened",
            len(ranked_candidates)
        )

        filter_col2.metric(
            "Matching Candidates",
            len(filtered_candidates)
        )

        filter_col3.metric(
            "Filtered Out",
            len(ranked_candidates) - len(filtered_candidates)
        )

        st.divider()

        if filtered_candidates:

            st.success(
                f"{len(filtered_candidates)} candidate(s) match your filters."
            )

            filtered_rows = []

            for rank, candidate in enumerate(
                filtered_candidates,
                start=1
            ):

                candidate_status = st.session_state[
                    "candidate_statuses"
                ].get(
                    candidate["file_name"],
                    "Under Review"
                )
                candidate_result = candidate.get("result") or {}

                if not isinstance(candidate_result, dict):
                    candidate_result = {}

                contact_data = candidate_result.get("contact_data") or {}

                if not isinstance(contact_data, dict):
                    contact_data = {}

                filtered_rows.append({

                    "Rank": rank,

                    "Candidate Name": candidate["candidate_name"],

                    "Resume": candidate["file_name"],

                    "Email": ", ".join(contact_data.get("emails") or []),
                    "Phone": ", ".join(contact_data.get("phone_numbers") or []),
                    "Location": ", ".join(contact_data.get("locations") or []),

                    "Skill Match (%)": candidate["match_percentage"],

                    "Recruiter Status": candidate_status

                })

            filtered_df = pd.DataFrame(filtered_rows)

            st.dataframe(
                filtered_df,
                width="stretch",
                hide_index=True
            )

            # ----------------------------------
            # EXPORT FILTERED CANDIDATES
            # ----------------------------------

            export_col1, export_col2 = st.columns(2)

            with export_col1:

                filtered_csv = filtered_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "📄 Download Filtered CSV",
                    data=filtered_csv,
                    file_name="HireMind_Filtered_Candidates.csv",
                    mime="text/csv",
                    key="filtered_candidates_csv",
                    width="stretch"
                )

            with export_col2:

                filtered_excel_buffer = BytesIO()

                with pd.ExcelWriter(
                    filtered_excel_buffer,
                    engine="openpyxl"
                ) as writer:

                    filtered_df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Filtered Candidates"
                    )

                st.download_button(
                    "📊 Download Filtered Excel",
                    data=filtered_excel_buffer.getvalue(),
                    file_name="HireMind_Filtered_Candidates.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="filtered_candidates_excel",
                    width="stretch"
                )

        else:

            st.warning(
                "No candidates match the selected filters. "
                "Try changing your search or filter conditions."
            )


        # ==================================
        # TOP METRICS
        # ==================================

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Resumes",
            len(results)
        )

        col2.metric(
            "Successfully Screened",
            len(successful_results)
        )

        col3.metric(
            "Failed Resumes",
            len(failed_results)
        )

        col4.metric(
            "Highest Skill Match",
            f"{ranked_candidates[0]['match_percentage']:.1f}%"
        )

        # ==================================
        # CANDIDATE COMPARISON
        # ==================================

        st.subheader("Candidate Comparison")

        ranking_rows = []

        for rank, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            candidate_status = st.session_state[
                "candidate_statuses"
            ].get(
                candidate["file_name"],
                "Under Review"
            )

            candidate_result = candidate.get("result") or {}

            if not isinstance(candidate_result, dict):
                candidate_result = {}

            candidate_data = candidate_result.get("skill_data") or {}

            if not isinstance(candidate_data, dict):
                candidate_data = {}

            matched_for_table = candidate_data.get("matched_skills") or []
            missing_for_table = candidate_data.get("missing_skills") or []

            if not isinstance(matched_for_table, list):
                matched_for_table = []

            if not isinstance(missing_for_table, list):
                missing_for_table = []

            ranking_rows.append({

                "Rank": rank,

                "Candidate Name": candidate.get(
                    "candidate_name",
                    "Not detected"
                ),

                "Resume": candidate.get(
                    "file_name",
                    ""
                ),

                "Matched Skills": ", ".join(
                    str(skill) for skill in matched_for_table
                    if isinstance(skill, str)
                ),

                "Missing Skills": ", ".join(
                    str(skill) for skill in missing_for_table
                    if isinstance(skill, str)
                ),

                "Skill Match (%)": candidate.get(
                    "match_percentage",
                    0
                ),

                "OCR Pages": candidate.get(
                    "ocr_pages",
                    0
                ),

                "Processing Method": candidate.get(
                    "processing_method",
                    "Unknown"
                ),

                "Recruiter Status": candidate_status

            })


        # Create DataFrame AFTER the loop
        ranking_df = pd.DataFrame(ranking_rows)


        # Display ONE combined comparison table
        if not ranking_df.empty:

            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True,
                key="candidate_comparison_table"
            )

        else:

            st.info("No candidates available for comparison.")

        # ==================================
        # CANDIDATE CONTACT INTELLIGENCE
        # ==================================

        st.divider()

        st.header("📇 Candidate Contact Intelligence")

        contact_rows = []
        for candidate in successful_results:
            candidate_result = candidate.get("result") or {}

            if not isinstance(candidate_result, dict):
                candidate_result = {}

            contact = candidate_result.get("contact_data") or {}

            if not isinstance(contact, dict):
                contact = {}

            contact_rows.append({

                "Candidate Name": candidate["candidate_name"],

                "Resume": candidate["file_name"],

                "Email": ", ".join(
                    contact.get("emails") or []
                ),

                "Phone": ", ".join(
                    contact.get("phone_numbers") or []
                ),

                "Location": ", ".join(
                    contact.get("locations") or []
                ),

               "LinkedIn": (
                    contact.get("linkedin_urls") or [""]
                )[0],

                "GitHub": (
                    contact.get("github_urls") or [""]
                )[0],

                "Portfolio": (
                    contact.get("portfolio_urls") or [""]
                )[0],

                "Recruiter Status": st.session_state[
                    "candidate_statuses"
                ].get(
                    candidate["file_name"],
                    "Under Review"
                )

            })

        contact_df = pd.DataFrame(contact_rows)

        # Candidate search
        contact_search = st.text_input(
            "🔍 Search candidates by name, email, phone or location",
            key="contact_directory_search"
        )

        filtered_contact_df = contact_df.copy()

        if contact_search.strip():

            search_value = contact_search.strip()

            matching_rows = filtered_contact_df.apply(
                lambda row: row.astype(str).str.contains(
                    search_value,
                    case=False,
                    regex=False
                ).any(),
                axis=1
            )

            filtered_contact_df = filtered_contact_df[
                matching_rows
            ]

        st.caption(
            f"Showing {len(filtered_contact_df)} of "
            f"{len(contact_df)} candidates"
        )

        st.dataframe(
            filtered_contact_df,
            width="stretch",
            hide_index=True,
            column_config={

                "LinkedIn": st.column_config.LinkColumn(
                    "LinkedIn"
                ),

                "GitHub": st.column_config.LinkColumn(
                    "GitHub"
                ),

                "Portfolio": st.column_config.LinkColumn(
                    "Portfolio"
                )

            }
        )

        # Download contact directory
        contact_excel_buffer = BytesIO()

        with pd.ExcelWriter(
            contact_excel_buffer,
            engine="openpyxl"
        ) as writer:

            contact_df.to_excel(
                writer,
                index=False,
                sheet_name="Candidate Contacts"
            )

        st.download_button(
            label="📥 Download Candidate Contact Directory",
            data=contact_excel_buffer.getvalue(),
            file_name="HireMind_Candidate_Contacts.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            key="download_candidate_contacts",
            width="stretch"
        )

        # ==================================
        # EXPORT REPORTS
        # ==================================

        st.divider()

        st.subheader("📥 Download Recruitment Report")

        col1, col2 = st.columns(2)

        with col1:

            csv_data = ranking_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📄 Download CSV Report",
                data=csv_data,
                file_name="HireMind_Candidate_Ranking.csv",
                mime="text/csv",
                width="stretch"
            )

        with col2:

            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                ranking_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Candidate Ranking"
                )

            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer.getvalue(),
                file_name="HireMind_Candidate_Ranking.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )

        # ==================================
        # CANDIDATE RANKING CHART
        # ==================================

        st.divider()

        st.subheader("📊 Candidate Skill Match Comparison")

        chart_df = pd.DataFrame([

            {
                "Candidate": (
                    candidate["candidate_name"]
                    if candidate["candidate_name"] != "Not detected"
                    else candidate["file_name"]
                ),

                "Skill Match (%)": candidate["match_percentage"]

            }

            for candidate in ranked_candidates

        ])

        fig = px.bar(
            chart_df,
            x="Skill Match (%)",
            y="Candidate",
            orientation="h",
            text="Skill Match (%)",
            color="Skill Match (%)",
            color_continuous_scale="Greens",
            title="Candidate Ranking by Skill Match"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_range=[0, 100]
        )
        st.plotly_chart(
            fig,
            width="stretch",
            key="candidate_ranking_comparison_chart"
        )

        # ==================================
        # HIGHEST MATCH CANDIDATE
        # ==================================

        best_candidate = ranked_candidates[0]

        st.divider()

        st.subheader("🥇 Highest Skill Match Candidate")

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Candidate: {best_candidate['candidate_name']}"
            )

            st.info(
                f"Resume: {best_candidate['file_name']}"
            )

        with col2:

            st.metric(
                "Skill Match",
                f"{best_candidate['match_percentage']:.2f}%"
            )

 
        # ==================================
        # INDIVIDUAL CANDIDATE REPORTS
        # ==================================

        st.divider()

        st.header("Individual Candidate Reports")

        for rank, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            candidate_result = candidate.get("result") or {}

            if not isinstance(candidate_result, dict):
                candidate_result = {}

            candidate_data = candidate_result.get("skill_data") or {}

            if not isinstance(candidate_data, dict):
                candidate_data = {}

            required = candidate_data.get(
                "required_skills", []
            ) or []

            matched = candidate_data.get(
                "matched_skills", []
            ) or []

            missing = candidate_data.get(
                "missing_skills", []
            ) or []

            total_skills = len(required)

            matched_count = len(matched)

            missing_count = len(missing)

            candidate_key = candidate.get(
                "file_name",
                f"candidate_{rank}"
            )

            # Unique key for charts belonging to this candidate
            chart_candidate_key = f"{rank}_{candidate_key}"

            with st.expander(
                f"Rank {rank} – {candidate.get('candidate_name', 'Unknown Candidate')} "
                f"({candidate.get('match_percentage', 0):.1f}%)"
            ):

                # ----------------------------------
                # RECRUITER DECISION
                # ----------------------------------

                st.subheader("Recruiter Decision")

                status_options = [
                    "Under Review",
                    "Shortlisted",
                    "Rejected"
                ]

                current_status = st.session_state[
                    "candidate_statuses"
                ].get(
                    candidate_key,
                    "Under Review"
                )

                selected_status = st.selectbox(
                    "Select Candidate Status",
                    status_options,
                    index=status_options.index(current_status)
                    if current_status in status_options else 0,
                    key=f"candidate_status_{rank}"
                )

                if (
                    st.session_state["candidate_statuses"].get(candidate_key)
                    != selected_status
                ):

                    st.session_state[
                        "candidate_statuses"
                    ][candidate_key] = selected_status

                    st.rerun()

                st.divider()


                # ----------------------------------
                # CANDIDATE METRICS
                # ----------------------------------

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Matched Skills",
                    matched_count
                )

                col2.metric(
                    "Missing Skills",
                    missing_count
                )

                col3.metric(
                    "Skill Match",
                    f"{candidate['match_percentage']:.1f}%"
                )
                                # ----------------------------------
                # CANDIDATE CONTACT PROFILE
                # ----------------------------------

                st.divider()

                st.subheader("📇 Candidate Contact Profile")

                contact_data = candidate_result.get("contact_data") or {}

                if not isinstance(contact_data, dict):
                    contact_data = {}

                contact_col1, contact_col2 = st.columns(2)

                with contact_col1:

                    st.markdown("**Email Addresses**")

                    emails = contact_data.get("emails") or []

                    if emails:

                        for email in emails:
                            st.code(email)

                    else:
                        st.caption("No email address detected.")

                    st.markdown("**Phone Numbers**")

                    phones = contact_data.get("phone_numbers") or []

                    if phones:

                        for phone in phones:
                            st.code(phone)

                    else:
                        st.caption("No phone number detected.")

                    st.markdown("**Locations**")

                    locations = contact_data.get("locations") or []

                    if locations:

                        for location in locations:
                            st.write(f"📍 {location}")

                    else:
                        st.caption("No explicitly labelled location detected.")

                with contact_col2:

                    st.markdown("**LinkedIn Profiles**")
                    linkedin = contact_data.get("linkedin_urls") or []

                    if linkedin:

                        for url in linkedin:
                            st.markdown(f"[Open LinkedIn Profile]({url})")

                    else:
                        st.caption("No LinkedIn URL detected.")

                    st.markdown("**GitHub Profiles**")

                    github = contact_data.get("github_urls") or []

                    if github:

                        for url in github:
                            st.markdown(f"[Open GitHub Profile]({url})")

                    else:
                        st.caption("No GitHub URL detected.")

                    st.markdown("**Portfolio / Website**")

                    portfolios = contact_data.get("portfolio_urls") or []

                    if portfolios:

                        for url in portfolios:
                            st.markdown(f"[Open Website]({url})")

                    else:
                        st.caption("No portfolio URL detected.")

                st.caption(
                    "Contact details are extracted automatically from "
                    "resume text. Verify them before contacting candidates."
                )

                # ----------------------------------
                # PDF INFORMATION
                # ----------------------------------

                st.subheader("PDF Processing Information")

                st.metric(
                    "Pages Processed Using OCR",
                    candidate["ocr_pages"]
                )

                st.caption(
                    f"Processing method: "
                    f"{candidate['processing_method']}"
                )

                # ----------------------------------
                # CANDIDATE OVERVIEW
                # ----------------------------------

                st.subheader("Candidate Overview")

                education = (
                    candidate_data.get("candidate_education")
                    or "Not specified"
                )

                experience = candidate_data.get(
                    "candidate_experience_years"
                )

                experience_evidence = (
                    candidate_data.get(
                        "candidate_experience_evidence"
                    )
                    or "No professional employment evidence extracted."
                )

                experience_requirement = (
                    format_experience_requirement(
                        candidate_data.get(
                            "required_experience_requirement"
                        )
                    )
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.info(
                        f"**Education:** {education}"
                    )

                with col2:

                    if experience is not None:

                        st.info(
                            f"**Verified Experience:** "
                            f"{experience:g} years"
                        )

                    else:

                        st.warning(
                            "Experience duration requires verification."
                        )

                st.markdown(
                    "**Professional Employment Evidence**"
                )

                st.write(experience_evidence)

                st.caption(
                    f"Required experience: {experience_requirement}"
                )

                           # ----------------------------------
                # SKILL MATCHING ANALYSIS
                # ----------------------------------

                st.divider()

                st.subheader("Skill Matching Analysis")

                chart_col1, chart_col2 = st.columns(2)

                # Create a unique chart key for each candidate
                chart_candidate_key = re.sub(
                    r"[^a-zA-Z0-9_]",
                    "_",
                    f"{rank}_{candidate_key}"
                )

                with chart_col1:

                    if total_skills > 0:

                        donut_data = pd.DataFrame({

                            "Status": ["Matched", "Missing"],

                            "Count": [
                                matched_count,
                                missing_count
                            ]

                        })

                        fig = px.pie(
                            donut_data,
                            names="Status",
                            values="Count",
                            hole=0.60,
                            color="Status",
                            color_discrete_map={
                                "Matched": "#16A34A",
                                "Missing": "#EF4444"
                            },
                            title="Required Skill Coverage"
                        )

                        fig.update_traces(
                            textinfo="label+percent"
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch",
                            key=f"candidate_skill_donut_{chart_candidate_key}"
                        )

                    else:

                        st.warning(
                            "No required skills were extracted."
                        )

                with chart_col2:

                    if required:

                        matched_keys = {
                            normalize_skill(skill)
                            for skill in matched
                        }

                        skill_rows = []

                        for skill in required:

                            status = (
                                "Matched"
                                if normalize_skill(skill) in matched_keys
                                else "Missing"
                            )

                            skill_rows.append({

                                "Skill": skill,

                                "Count": 1,

                                "Status": status

                            })

                        skill_df = pd.DataFrame(skill_rows)

                        fig = px.bar(
                            skill_df,
                            x="Count",
                            y="Skill",
                            color="Status",
                            orientation="h",
                            title="Individual Skill Analysis",
                            color_discrete_map={
                                "Matched": "#16A34A",
                                "Missing": "#EF4444"
                            }
                        )

                        fig.update_layout(
                            yaxis={
                                "categoryorder": "total ascending"
                            },
                            xaxis_title="Requirement count",
                            yaxis_title=""
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch",
                            key=f"candidate_skill_bar_{chart_candidate_key}"
                        )
                # ----------------------------------
                # MATCHED AND MISSING SKILLS
                # ----------------------------------

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("✅ Matched Skills")

                    if matched:

                        for skill in matched:
                            st.success(skill)

                    else:

                        st.info(
                            "No matched skills identified."
                        )

                with col2:

                    st.subheader("❌ Missing Skills")

                    if missing:

                        for skill in missing:
                            st.error(skill)

                    else:

                        st.success(
                            "No missing skills identified."
                        )

                # ----------------------------------
                # SKILL EVIDENCE
                # ----------------------------------

                evidence = candidate_data.get(
                    "skill_evidence", []
                )

                if evidence:

                    st.divider()

                    st.subheader("Skill Evidence")

                    evidence_df = pd.DataFrame(evidence)

                    st.dataframe(
                        evidence_df,
                        width="stretch",
                        hide_index=True
                    )

                # ----------------------------------
                # EDUCATION ELIGIBILITY
                # ----------------------------------

                st.divider()

                st.subheader("Education Eligibility")

                education_status = candidate_data.get(
                    "education_meets_requirement"
                )

                if education_status is True:

                    st.success(
                        "Education requirement met."
                    )

                elif education_status is False:

                    st.warning(
                        "Education requirement not met."
                    )

                else:

                    st.info(
                        "Education eligibility needs verification."
                    )

                # ----------------------------------
                # EXPERIENCE ELIGIBILITY
                # ----------------------------------

                st.subheader("Experience Eligibility")

                experience_status = candidate_data.get(
                    "experience_meets_requirement"
                )

                if experience_status is True:

                    st.success(
                        "Experience requirement met."
                    )

                elif experience_status is False:

                    st.warning(
                        "Experience requirement not met."
                    )

                else:

                    st.info(
                        "Experience requirement cannot be confirmed."
                    )

                # ----------------------------------
                # AI ANALYSIS REPORTS
                # ----------------------------------

                st.divider()

                st.subheader("AI Analysis Reports")

                with st.expander(
                    "Resume Screening Report"
                ):

                    st.markdown(
                        candidate_result["screening_result"]
                    )

                with st.expander(
                    "Structured Skill Matching Data"
                ):

                    st.json(candidate_data)

                with st.expander(
                    "Candidate Evaluation Report"
                ):

                    st.markdown(
                        candidate_result["evaluation_result"]
                    )

    else:

        st.warning(
            "No resumes were successfully screened."
        )

    # ==================================
    # FAILED RESUMES
    # ==================================

    if failed_results:

        st.divider()

        st.subheader("Failed Resume Processing")

        for candidate in failed_results:

            st.warning(
                f"{candidate['file_name']}: "
                f"{candidate['error']}"
            )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "HireMind AI | Python | Streamlit | Groq | "
    "SQLite | PyMuPDF | Tesseract OCR | Plotly"
)
