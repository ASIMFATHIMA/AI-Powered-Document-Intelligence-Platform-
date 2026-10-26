
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 20

def validate_file(uploaded_file):
    if uploaded_file is None:
        return False, "Please select a document."

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Only PDF, DOCX, and TXT files are supported."

    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        return False, "File size must be less than 20 MB."

    if not uploaded_file.getvalue():
        return False, "The uploaded file is empty."

    return True, "File validated successfully."
