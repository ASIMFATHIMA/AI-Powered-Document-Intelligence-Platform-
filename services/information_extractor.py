
import json
import os
import re
from typing import Any

import ollama


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = os.getenv("OLLAMA_MODEL", "tinyllama")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_CHUNKS = 4
MAX_ATTEMPTS = 2

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "title": {"type": "string"},
        "people": {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "numbers_and_amounts": {
            "type": "array",
            "items": {"type": "string"}
        },
        "action_items": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "document_type",
        "title",
        "people",
        "organizations",
        "dates",
        "locations",
        "skills",
        "key_facts",
        "numbers_and_amounts",
        "action_items"
    ]
}


# --------------------------------------------------
# Default result
# --------------------------------------------------

def _empty_result() -> dict:
    return {
        "document_type": "Unknown",
        "title": "Untitled Document",
        "people": [],
        "organizations": [],
        "dates": [],
        "locations": [],
        "skills": [],
        "key_facts": [],
        "numbers_and_amounts": [],
        "action_items": []
    }


# --------------------------------------------------
# Text utilities
# --------------------------------------------------

def _unique(items: list) -> list:
    """Remove duplicate strings while preserving their order."""
    result = []
    seen = set()

    for item in items:
        if not isinstance(item, str):
            continue

        item = re.sub(r"\s+", " ", item).strip()

        if not item:
            continue

        key = item.casefold()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _document_chunks(text: str) -> list[str]:
    """Split long documents into smaller overlapping sections."""
    text = re.sub(r"\r\n?", "\n", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a sentence or paragraph boundary.
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end)
            )

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(end - CHUNK_OVERLAP, start + 1)
        start = next_start

    return chunks


def _normalize_result(data: Any) -> dict:
    """Normalize and validate the model's extracted fields."""
    result = _empty_result()

    if not isinstance(data, dict):
        return result

    result["document_type"] = str(
        data.get("document_type") or "Unknown"
    ).strip()

    result["title"] = str(
        data.get("title") or "Untitled Document"
    ).strip()

    list_fields = [
        "people",
        "organizations",
        "dates",
        "locations",
        "skills",
        "key_facts",
        "numbers_and_amounts",
        "action_items"
    ]

    for field in list_fields:
        value = data.get(field, [])

        if isinstance(value, list):
            result[field] = _unique([
                str(item) for item in value
                if isinstance(item, (str, int, float))
            ])
        elif isinstance(value, str) and value.strip():
            result[field] = [value.strip()]

    return result


# --------------------------------------------------
# Prompt construction
# --------------------------------------------------

def _build_prompt(text: str, retry: bool = False) -> str:
    instruction = """
You are a document information extraction assistant.

Read the document section and extract only information supported
by its text. Do not invent names, dates, organizations, skills,
facts, amounts, or action items.

Return one valid JSON object with exactly these fields:
- document_type: string
- title: string
- people: array of strings
- organizations: array of strings
- dates: array of strings
- locations: array of strings
- skills: array of strings
- key_facts: array of strings
- numbers_and_amounts: array of strings
- action_items: array of strings

Use an empty array when a field has no supported information.
Use "Unknown" for an unknown document type and
"Untitled Document" for an unknown title.

Do not include Markdown fences, explanations, or text outside JSON.
"""

    if retry:
        instruction += """
IMPORTANT: Ensure every JSON string is properly quoted and escaped.
Close every array, object, and string before finishing the response.
"""

    return (
        instruction
        + "\nDOCUMENT SECTION:\n"
        + text
        + "\n\nReturn the JSON object now."
    )


# --------------------------------------------------
# JSON parsing
# --------------------------------------------------

def _parse_response(response: str) -> dict:
    """Parse JSON returned by TinyLlama."""
    if not response or not response.strip():
        raise ValueError("The model returned an empty response.")

    content = response.strip()

    # Remove Markdown code fences if the model adds them.
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
    content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to recover a JSON object from surrounding text.
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end <= start:
            raise ValueError("No complete JSON object was returned.")

        data = json.loads(content[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("The response must be a JSON object.")

    return _normalize_result(data)


# --------------------------------------------------
# Rule-based fallback
# --------------------------------------------------

def _fallback_extract(text: str) -> dict:
    """
    Extract basic information without depending on valid LLM JSON.
    This fallback is heuristic and may miss information.
    """
    result = _empty_result()

    if not text.strip():
        return result

    lines = [
        line.strip(" \t-*•")
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        result["title"] = lines[0][:200]

    lower_text = text.lower()

    if any(word in lower_text for word in [
        "resume", "curriculum vitae", "work experience"
    ]):
        result["document_type"] = "Resume"

    elif any(word in lower_text for word in [
        "invoice", "amount due", "invoice number"
    ]):
        result["document_type"] = "Invoice"

    elif any(word in lower_text for word in [
        "research paper", "abstract", "methodology"
    ]):
        result["document_type"] = "Research Paper"

    elif any(word in lower_text for word in [
        "agreement", "hereby", "contract", "clause"
    ]):
        result["document_type"] = "Legal Document"

    elif any(word in lower_text for word in [
        "meeting minutes", "agenda", "action items"
    ]):
        result["document_type"] = "Meeting Notes"

    # Dates: common numeric and written date formats.
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b"
    ]

    dates = []

    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, flags=re.I))

    result["dates"] = _unique(dates)

    # Numbers and amounts.
    amount_pattern = (
        r"(?:₹|Rs\.?|INR|USD|\$|€|£)\s*"
        r"\d[\d,]*(?:\.\d{1,2})?"
        r"|\b\d+(?:\.\d+)?\s*%"
    )

    result["numbers_and_amounts"] = _unique(
        re.findall(amount_pattern, text, flags=re.I)
    )

    # Common technical skills. These are included only if mentioned.
    skill_terms = [
        "Python", "Java", "JavaScript", "SQL", "MySQL",
        "PostgreSQL", "Machine Learning", "Deep Learning",
        "Artificial Intelligence", "NLP", "Natural Language Processing",
        "Generative AI", "LLM", "RAG", "TensorFlow", "PyTorch",
        "Pandas", "NumPy", "Scikit-learn", "FastAPI", "Flask",
        "Django", "Streamlit", "React", "Node.js", "HTML", "CSS",
        "Docker", "Git", "GitHub", "Power BI", "Excel",
        "Data Analysis", "Communication", "Leadership",
        "Problem Solving", "Project Management"
    ]

    found_skills = []

    for skill in skill_terms:
        if re.search(
            r"(?<!\w)" + re.escape(skill) + r"(?!\w)",
            text,
            flags=re.I
        ):
            found_skills.append(skill)

    result["skills"] = _unique(found_skills)

    # Conservative fallback for people: only look for explicit labels.
    for line in lines:
        match = re.match(
            r"^(?:Name|Candidate Name|Prepared by|Author)\s*:\s*(.+)$",
            line,
            flags=re.I
        )
        if match:
            result["people"].append(match.group(1).strip())

    result["people"] = _unique(result["people"])

    # Use short, non-empty lines as basic facts if no LLM result exists.
    result["key_facts"] = _unique([
        line[:300] for line in lines
        if len(line) >= 25
    ])[:10]

    return result


# --------------------------------------------------
# LLM extraction
# --------------------------------------------------

def _extract_chunk(text: str, section_number: int) -> dict:
    """Ask TinyLlama to extract information from one section."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": _build_prompt(
                            text,
                            retry=(attempt > 0)
                        )
                    }
                ],
                format=OUTPUT_SCHEMA,
                options={
                    "temperature": 0,
                    "num_predict": 1400,
                    "num_ctx": 4096
                }
            )

            content = response.message.content
            return _parse_response(content)

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

        except Exception as exc:
            # Do not hide model connection or installation errors.
            raise RuntimeError(
                f"TinyLlama/Ollama error in section {section_number}: {exc}"
            ) from exc

    # Avoid failing the entire document because one section produced
    # malformed or incomplete JSON.
    print(
        f"Warning: section {section_number} returned invalid JSON: "
        f"{last_error}. Using rule-based fallback."
    )

    return _fallback_extract(text)


# --------------------------------------------------
# Merge section results
# --------------------------------------------------

def _merge_results(results: list[dict]) -> dict:
    """Combine extracted information from all document sections."""
    merged = _empty_result()

    list_fields = [
        "people",
        "organizations",
        "dates",
        "locations",
        "skills",
        "key_facts",
        "numbers_and_amounts",
        "action_items"
    ]

    for result in results:
        if (
            merged["document_type"] == "Unknown"
            and result.get("document_type") != "Unknown"
        ):
            merged["document_type"] = result["document_type"]

        if (
            merged["title"] == "Untitled Document"
            and result.get("title") != "Untitled Document"
        ):
            merged["title"] = result["title"]

        for field in list_fields:
            merged[field].extend(result.get(field, []))

    for field in list_fields:
        merged[field] = _unique(merged[field])

    return merged


# --------------------------------------------------
# Public function called by app.py
# --------------------------------------------------

def extract_information(document_text: str) -> dict:
    """
    Extract structured information from document text.

    IMPORTANT:
    Call this function with ONE argument:
        extract_information(text)
    """
    if not isinstance(document_text, str):
        raise TypeError("Document text must be a string.")

    if not document_text.strip():
        raise ValueError(
            "The document contains no readable text. "
            "Please upload a text-based PDF, DOCX, or TXT file."
        )

    chunks = _document_chunks(document_text)

    if not chunks:
        return _empty_result()

    extracted_sections = []

    for index, chunk in enumerate(chunks, start=1):
        extracted_sections.append(
            _extract_chunk(chunk, index)
        )

    return _merge_results(extracted_sections)
