
import json
import os
from pathlib import Path

import streamlit as st

from database.db import (
    init_db,
    save_document,
    save_feedback,
    recent_documents,
)
from utils.file_validator import validate_file
from services.document_processor import extract_text
from services.summarizer import summarize_document, explain_document
from services.information_extractor import extract_information
from services.adaptive_analyzer import (
    analyze_document,
    DOCUMENT_MODES,
)
from services.document_chat import ask_document
from services.visualizer import (
    keyword_counts,
    find_numbers,
    make_keyword_chart,
    make_numeric_chart,
)


# --------------------------------------------------
# 1. Project directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI-Powered Document Intelligence Platform")

st.write(
    "Upload a document to summarize it, extract information, "
    "analyze findings, visualize data, and ask questions."
)


# --------------------------------------------------
# 3. Initialize database safely
# --------------------------------------------------

try:
    init_db()
    db_ready = True
except Exception as exc:
    db_ready = False
    st.error(f"Database initialization failed: {exc}")
    st.info(
        "Check your MySQL server, database name, and credentials."
    )


# --------------------------------------------------
# 4. Initialize session state
# --------------------------------------------------

defaults = {
    "document_text": "",
    "document_id": None,
    "document_name": "",
    "chat_history": [],
    "results": {},
}

for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --------------------------------------------------
# 5. Sidebar and model configuration
# --------------------------------------------------

st.sidebar.title("📄 DocuMind AI")
st.sidebar.caption("AI-Powered Document Intelligence")

model = st.sidebar.text_input(
    "Ollama model",
    value=os.getenv("OLLAMA_MODEL", "tinyllama"),
    help="Use the model name shown by the ollama list command.",
).strip()

if not model:
    model = "tinyllama"

st.sidebar.caption(f"Selected model: {model}")

st.sidebar.subheader("Recent Documents")

if db_ready:
    try:
        documents = recent_documents()

        if documents:
            for item in documents:
                st.sidebar.caption(
                    f"#{item['id']} · {item['filename']} · "
                    f"{item['text_length']:,} characters"
                )
        else:
            st.sidebar.caption("No recent documents.")

    except Exception as exc:
        st.sidebar.warning(f"Could not load recent documents: {exc}")


# --------------------------------------------------
# 6. Upload and process document
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX, or TXT",
    type=["pdf", "docx", "txt"],
)

if uploaded_file:
    valid, message = validate_file(uploaded_file)

    if not valid:
        st.error(message)

    else:
        st.caption(message)

        if st.button("Process Document", type="primary"):
            try:
                with st.spinner("Reading document..."):
                    extracted_text = extract_text(uploaded_file)

                if not extracted_text or not extracted_text.strip():
                    st.error(
                        "No readable text was found. "
                        "Check whether the document is scanned."
                    )

                else:
                    safe_name = Path(uploaded_file.name).name
                    file_path = UPLOAD_DIR / safe_name

                    file_path.write_bytes(uploaded_file.getvalue())

                    new_document_id = None

                    if db_ready:
                        try:
                            new_document_id = save_document(
                                uploaded_file.name,
                                len(extracted_text),
                            )
                        except Exception as exc:
                            st.warning(
                                f"Document could not be saved to MySQL: {exc}"
                            )

                    st.session_state.document_text = extracted_text
                    st.session_state.document_id = new_document_id
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.results = {}

                    st.success("Document processed successfully!")

            except Exception as exc:
                st.error(f"Document processing failed: {exc}")


# --------------------------------------------------
# 7. Display active document
# --------------------------------------------------

text = st.session_state.document_text

if text:
    st.success(
        f"Active document: {st.session_state.document_name} "
        f"({len(text):,} characters)"
    )

    with st.expander("Preview Extracted Text"):
        st.text_area(
            "Document content",
            value=text[:15000],
            height=220,
            disabled=True,
        )

    tabs = st.tabs(
        [
            "Summary",
            "Information Extraction",
            "Adaptive Analysis",
            "Visual Insights",
            "Document Chat",
            "Feedback & Export",
        ]
    )

    # --------------------------------------------------
    # TAB 1: Summary and explanation
    # --------------------------------------------------

    with tabs[0]:
        st.subheader("Document Summary")

        style = st.selectbox(
            "Summary detail",
            ["Short", "Balanced", "Detailed"],
        )

        if st.button("Generate Summary"):
            try:
                with st.spinner("Generating summary..."):
                    result = summarize_document(text, style, model)

                st.session_state.results["summary"] = result

            except Exception as exc:
                st.error(f"Summary error: {exc}")

        summary = st.session_state.results.get("summary")

        if summary:
            st.markdown(summary)

            st.download_button(
                "Download Summary",
                data=str(summary),
                file_name="summary.txt",
                mime="text/plain",
            )

        st.divider()

        st.subheader("Simple English Explanation")

        if st.button("Explain in Simple English"):
            try:
                with st.spinner("Generating explanation..."):
                    result = explain_document(text, model)

                st.session_state.results["explanation"] = result

            except Exception as exc:
                st.error(f"Explanation error: {exc}")

        explanation = st.session_state.results.get("explanation")

        if explanation:
            st.markdown(explanation)

            st.download_button(
                "Download Explanation",
                data=str(explanation),
                file_name="explanation.txt",
                mime="text/plain",
            )

    # --------------------------------------------------
    # TAB 2: Structured information extraction
    # --------------------------------------------------

    with tabs[1]:
        st.subheader("Structured Information Extraction")

        st.caption(
            "Uses your local Ollama model. "
            "Rule-based fallback may be used if model output is invalid."
        )

        if st.button("Extract Structured Information"):
            try:
                with st.spinner("Extracting information..."):
                    extracted = extract_information(text)

                if not isinstance(extracted, dict):
                    st.error(
                        "The extraction service did not return a JSON object."
                    )
                    st.session_state.results.pop("extracted", None)

                else:
                    st.session_state.results["extracted"] = extracted
                    st.success("Information extraction completed.")

            except Exception as exc:
                st.error(f"Extraction error: {exc}")

        extracted = st.session_state.results.get("extracted")

        if extracted and isinstance(extracted, dict):
            st.json(extracted)

            st.download_button(
                "Download JSON",
                data=json.dumps(
                    extracted,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="extracted_information.json",
                mime="application/json",
            )

            rows = []

            for key, value in extracted.items():
                if isinstance(value, list):
                    display_value = ", ".join(
                        str(item) for item in value
                    )
                elif isinstance(value, dict):
                    display_value = json.dumps(
                        value, ensure_ascii=False
                    )
                else:
                    display_value = str(value)

                rows.append(
                    {
                        "Field": key,
                        "Value": display_value,
                    }
                )

            st.dataframe(
                rows,
                use_container_width=True,
            )

    # --------------------------------------------------
    # TAB 3: Adaptive document analysis
    # --------------------------------------------------

    with tabs[2]:
        st.subheader("Adaptive Document Analysis")

        mode = st.selectbox(
            "Select Document Category",
            list(DOCUMENT_MODES.keys()),
        )

        if st.button("Analyze Document"):
            try:
                with st.spinner("Analyzing document..."):
                    result = analyze_document(text, mode, model)

                st.session_state.results["analysis"] = result

            except Exception as exc:
                st.error(f"Analysis error: {exc}")

        analysis = st.session_state.results.get("analysis")

        if analysis:
            st.markdown(analysis)

            st.download_button(
                "Download Analysis",
                data=str(analysis),
                file_name="analysis.txt",
                mime="text/plain",
            )

    # --------------------------------------------------
    # TAB 4: Visual insights
    # --------------------------------------------------

    with tabs[3]:
        st.subheader("Keyword Analysis")

        try:
            keywords = keyword_counts(text)

            if keywords is not None and not keywords.empty:
                keyword_chart = make_keyword_chart(keywords)

                st.plotly_chart(
                    keyword_chart,
                    use_container_width=True,
                )

                st.dataframe(
                    keywords,
                    use_container_width=True,
                )
            else:
                st.info("No keywords found.")

        except Exception as exc:
            st.warning(f"Keyword chart error: {exc}")

        st.divider()
        st.subheader("Numerical Insights")

        try:
            numbers = find_numbers(text)

            if numbers is not None and not numbers.empty:
                numeric_chart = make_numeric_chart(numbers)

                st.plotly_chart(
                    numeric_chart,
                    use_container_width=True,
                )

                st.dataframe(
                    numbers,
                    use_container_width=True,
                )

                st.download_button(
                    "Download Numerical Data CSV",
                    data=numbers.to_csv(index=False),
                    file_name="numerical_insights.csv",
                    mime="text/csv",
                )

            else:
                st.info("No numeric values found.")

        except Exception as exc:
            st.warning(f"Numerical chart error: {exc}")

        st.caption(
            "Charts use heuristic extraction. Verify the values "
            "against the original document."
        )

    # --------------------------------------------------
    # TAB 5: Document chatbot
    # --------------------------------------------------

    with tabs[4]:
        st.subheader("💬 Ask Questions About Your Document")

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

        question = st.chat_input(
            "Ask a question about your document..."
        )

        if question:
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Finding an answer..."):
                        answer = ask_document(
                            text,
                            question,
                            st.session_state.chat_history[:-1],
                            model,
                        )

                    st.markdown(answer)

            except Exception as exc:
                answer = f"Chat error: {exc}"

                with st.chat_message("assistant"):
                    st.error(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": str(answer),
                }
            )

    # --------------------------------------------------
    # TAB 6: Feedback and export
    # --------------------------------------------------

    with tabs[5]:
        st.subheader("Share Your Feedback")

        rating = st.radio(
            "Was this result helpful?",
            ["Helpful", "Not helpful"],
            horizontal=True,
        )

        comment = st.text_area(
            "Additional comments (optional)"
        )

        if st.button("Submit Feedback"):
            if not db_ready:
                st.error(
                    "MySQL is unavailable. Feedback could not be saved."
                )

            elif st.session_state.document_id is None:
                st.error(
                    "The document has no database ID. "
                    "Please check your MySQL connection and process it again."
                )

            else:
                try:
                    save_feedback(
                        st.session_state.document_id,
                        rating,
                        comment,
                    )

                    st.success("Feedback saved successfully.")

                except Exception as exc:
                    st.error(f"Feedback error: {exc}")

        st.divider()

        st.download_button(
            "Download Extracted Text",
            data=text,
            file_name="extracted_text.txt",
            mime="text/plain",
        )

        if st.session_state.results:
            st.download_button(
                "Download All Results as JSON",
                data=json.dumps(
                    st.session_state.results,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                file_name="document_results.json",
                mime="application/json",
            )

else:
    st.info(
        "Upload a document and click Process Document to begin."
    )

    st.markdown(
        """
        ### Features

        - AI summaries and simple explanations
        - Structured information extraction
        - Adaptive document analysis
        - Interactive keyword and numerical charts
        - Document-aware chatbot
        - Feedback collection and downloads
        """
    )
