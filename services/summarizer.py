
from ollama import chat


def generate_response(prompt, model="qwen3:8b"):
    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2}
    )

    return response.message.content.strip()


def summarize_document(text, style="Balanced", model="qwen3:8b"):
    instructions = {
        "Short": "Summarize in 5 to 7 bullet points.",
        "Balanced": "Provide an overview, key points, and conclusion.",
        "Detailed": "Provide a detailed summary with headings and examples."
    }

    prompt = f"""
    Summarize the following document in clear English.

    Instructions: {instructions.get(style, instructions["Balanced"])}

    Use only facts supported by the document.

    DOCUMENT:
    {text[:20000]}
    """

    return generate_response(prompt, model)


def explain_document(text, model="qwen3:8b"):
    prompt = f"""
    Explain this document in simple English for a beginner.
    Define difficult terms, explain the main ideas, and list key takeaways.
    Do not invent information.

    DOCUMENT:
    {text[:18000]}
    """

    return generate_response(prompt, model)
