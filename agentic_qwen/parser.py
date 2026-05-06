"""
parser.py — LlamaParse PDF-parsing helper.

Kept as a standalone module so the agent can import it independently and so
that the Streamlit UI can call it directly without going through LangGraph.
"""

import os
from typing import Any

from tools import _extract_all_images_from_pdf   # reuse PyMuPDF fallback


# ──────────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> tuple[str, list[dict]]:
    """
    Parse *pdf_path* with LlamaParse and return (markdown_content, image_metadata).

    Falls back to direct PyMuPDF image extraction when LlamaParse returns no
    image metadata (e.g. the document has no embedded figure annotations).

    Raises:
        ValueError: if LLAMA_PARSE_API_KEY is not set.
        RuntimeError: if LlamaParse returns no documents.
    """
    from llama_parse import LlamaParse

    api_key = os.environ.get("LLAMA_PARSE_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_PARSE_API_KEY environment variable not set.")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parse_image_metadata=True,
    )

    documents = parser.load_data(pdf_path)
    if not documents:
        raise RuntimeError(f"LlamaParse returned no documents for: {pdf_path}")

    markdown_content = ""
    image_metadata: list[dict] = []

    for doc in documents:
        markdown_content += doc.text
        if doc.metadata and "image_metadata" in doc.metadata:
            image_metadata.extend(doc.metadata["image_metadata"])

    # Fallback when LlamaParse omits image metadata
    if not image_metadata:
        image_metadata = _extract_all_images_from_pdf(pdf_path)

    return markdown_content, image_metadata