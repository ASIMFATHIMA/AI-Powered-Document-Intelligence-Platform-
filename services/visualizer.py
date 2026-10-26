
import re
from collections import Counter
import pandas as pd
import plotly.express as px


def keyword_counts(text, top_n=12):
    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]{3,}\b",
        text.lower()
    )

    stopwords = {
        "this", "that", "with", "from", "have",
        "were", "will", "your", "their", "there",
        "which", "when", "where", "what", "about",
        "into", "also", "been", "being", "these",
        "those", "should", "would", "could"
    }

    counts = Counter(
        word for word in words if word not in stopwords
    )

    return pd.DataFrame(
        counts.most_common(top_n),
        columns=["Keyword", "Count"]
    )


def find_numbers(text):
    rows = []

    for line in text.splitlines():
        matches = re.findall(
            r"(?<!\w)(?:₹|\$|€)?\s*-?\d+(?:,\d{3})*(?:\.\d+)?%?",
            line
        )

        for match in matches:
            value = re.sub(r"[₹$€,%\s]", "", match)

            try:
                rows.append({
                    "Source": line[:150],
                    "Value": float(value),
                    "Original": match.strip()
                })
            except ValueError:
                continue

    return pd.DataFrame(
        rows, columns=["Source", "Value", "Original"]
    )


def make_keyword_chart(data):
    if data.empty:
        return None

    return px.bar(
        data,
        x="Keyword",
        y="Count",
        title="Most Frequent Keywords"
    )


def make_numeric_chart(data):
    if data.empty:
        return None

    chart_data = data.head(25).copy()
    chart_data["Item"] = [
        f"Item {i + 1}" for i in range(len(chart_data))
    ]

    return px.bar(
        chart_data,
        x="Item",
        y="Value",
        hover_data=["Source", "Original"],
        title="Numbers Found in Document"
    )
