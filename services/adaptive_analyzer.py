import ollama

MODEL_NAME = "tinyllama"

DOCUMENT_MODES = {
    "General Document": (
        "Give a balanced overview of the document's purpose, main points, "
        "important evidence, and practical implications."
    ),
    "Research Paper": (
        "Focus on the research question, approach, reported findings, "
        "evidence, limitations, and conclusions."
    ),
    "Business Report": (
        "Focus on objectives, reported results, business implications, "
        "risks, decisions, and recommended actions."
    ),
    "Legal Document": (
        "Focus on the parties, stated obligations, rights, conditions, "
        "deadlines, and potential ambiguities. Describe the text without "
        "giving legal advice."
    ),
    "Resume / CV": (
        "Focus on the candidate's stated profile, experience, skills, "
        "achievements, education, and areas that are not specified."
    ),
    "Financial Document": (
        "Focus on reporting periods, stated amounts, financial results, "
        "trends explicitly supported by the text, and disclosed risks. "
        "Do not calculate or infer unstated values."
    ),
    "Meeting Notes": (
        "Focus on attendees, discussion topics, decisions, action items, "
        "owners, and due dates when they are stated."
    ),
    "Technical Document": (
        "Focus on the system or subject, requirements, components, "
        "procedures, constraints, and technical risks."
    ),
}


def analyze_document(
    document_text: str,
    analysis_mode: str = "General Document",
    model: str = MODEL_NAME,
) -> str:
    """Analyze document content using the selected document category."""

    if not document_text or not document_text.strip():
        return "No document text was provided. Please upload a readable document."

    if analysis_mode not in DOCUMENT_MODES:
        raise ValueError(f"Unsupported document analysis mode: {analysis_mode}")

    if not model or not model.strip():
        raise ValueError("An Ollama model name must be provided.")

    document_text = document_text.strip()[:6000]
    analysis_focus = DOCUMENT_MODES[analysis_mode]

    prompt = f"""
Analyze the actual document below.

Analysis type: {analysis_mode}
Analysis focus: {analysis_focus}

Write a useful report with exactly these eight headings:

1. Document Overview
2. Main Findings
3. Important Details
4. Missing or Unclear Information
5. Document Structure
6. Importance
7. Recommendations or Next Steps
8. Conclusion

STRICT RULES:
- Analyze the document content, not these instructions.
- Do not repeat the headings or duplicate sections.
- Write 2 to 4 concise bullet points under each heading.
- Include actual names, facts, dates, numbers, and examples only when present.
- If information is absent, explicitly say "Not specified in the document."
- Do not invent statistics, references, case studies, or conclusions.
- Keep the language simple and professional.
- Do not explain what each heading means.
- Do not repeat the same recommendation in multiple sections.
- Return the report directly in Markdown.

DOCUMENT CONTENT:
<<<
{document_text}
>>>

ANALYSIS REPORT:
"""

    response = ollama.chat(
        model=model.strip(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze document content accurately. "
                    "Never copy instructions as the answer. "
                    "Do not invent facts or repeat sections."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 1000
        }
    )

    result = response.message.content.strip()

    if not result:
        raise RuntimeError("The model returned an empty analysis.")

    return result
