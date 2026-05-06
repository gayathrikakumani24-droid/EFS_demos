"""
tools.py — LangChain Tool definitions for the PDF → Qwen Vision agentic workflow.

Each tool wraps one atomic step of the original pipeline.
State is passed through the shared WorkflowState TypedDict (defined in agents.py).
"""

import os
import base64
import re
import json
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.tools import tool

# ──────────────────────────────────────────────────────────────────────────────
# Constants (overridable via env vars at runtime)
# ──────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
QWEN_VISION_MODEL   = "qwen/qwen2.5-vl-72b-instruct"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — PDF → Markdown (LlamaParse)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def parse_pdf_tool(pdf_path: str) -> dict:
    """
    Parse a PDF file with LlamaParse and return:
      - markdown_content (str)
      - image_metadata   (list[dict])

    Falls back to direct PyMuPDF image extraction when LlamaParse returns no
    image metadata.
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

    markdown_content = ""
    image_metadata: list[dict] = []

    for doc in documents:
        markdown_content += doc.text
        if doc.metadata and "image_metadata" in doc.metadata:
            image_metadata.extend(doc.metadata["image_metadata"])

    # Fallback: extract images directly with PyMuPDF
    if not image_metadata:
        image_metadata = _extract_all_images_from_pdf(pdf_path)

    return {
        "markdown_content": markdown_content,
        "image_metadata": image_metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Extract a single image from the PDF
# ══════════════════════════════════════════════════════════════════════════════

@tool
def extract_image_tool(payload: dict) -> str:
    """
    Extract one image from a PDF using metadata.
    """

    pdf_path   = payload["pdf_path"]
    image_info = payload["image_info"]
    output_dir = payload.get("output_dir", tempfile.gettempdir())

    return _extract_image_from_pdf(pdf_path, image_info, output_dir)
# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Describe an image with Qwen Vision (via OpenRouter)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def describe_image_tool(payload: dict) -> str:
    """
    Send an image to the Qwen Vision model on OpenRouter together with
    optional markdown context and extracted protocol rules.

    payload keys:
      - image_path   (str)
      - context_text (str, optional)
      - rules_text   (str, optional)

    Returns the model's description string.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable not set.")

    image_path   = payload["image_path"]
    context_text = payload.get("context_text", "")
    rules_text   = payload.get("rules_text", "")

    client = OpenAI(base_url=OPENROUTER_API_BASE, api_key=api_key)

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""
You are a digital protocol analyst.

You are given:
- An image (diagram)
- Context from a technical document
- Extracted protocol rules

-------------------------------------

Context:
{context_text}

Rules:
{rules_text}

-------------------------------------

Step 1: Identify diagram type:

Choose ONE:
1. Timing Diagram → signals vs time (waveforms, clock edges)
2. Dependency Diagram → arrows between signals (graph)
3. Simple Diagram → labels, blocks, or minimal relationships

-------------------------------------

Step 2: Analyze based on type:

IF Timing Diagram:
- Identify signals
- Describe signal transitions (HIGH/LOW)
- Explain when interactions happen
- Identify handshake events

IF Dependency Diagram:
The diagram may contain small arrows and labels.
Carefully zoom in mentally and trace each arrow precisely from source to destination.
Do not ignore thin or curved arrows.
- Treat as directed graph
- Nodes = signals
- Arrows = dependencies (A → B means B depends on A)
- If multiple arrows → AND condition

IF Simple Diagram:
- Identify signals or components
- Describe relationships in plain terms
- Do NOT overcomplicate

-------------------------------------

Step 3: Apply rules (if relevant)

-------------------------------------
FINAL INSTRUCTION (VERY IMPORTANT):

- Output ONLY 2–3 concise lines
- Do NOT include step numbers
- Do NOT include sections like "Step 1", "Signals", etc.
- Do NOT explain everything — summarize only key interaction
- Maximum 50 words

Return ONLY the final answer.
"""

    response = client.chat.completions.create(
        model=QWEN_VISION_MODEL,
        max_tokens=512,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Splice Qwen descriptions back into markdown
# ══════════════════════════════════════════════════════════════════════════════

@tool
def splice_descriptions_tool(markdown_content: str, image_descriptions: list[str]) -> str:
    """
    Insert image descriptions into markdown at figure references.
    Falls back to appending all descriptions at the end if no references found.
    """
    if not image_descriptions:
        return markdown_content

    figure_pattern = r"(Figure\s+[A-Za-z0-9\.\-:]+|Fig\.\s*[A-Za-z0-9\.\-:]+)"
    matches = list(re.finditer(figure_pattern, markdown_content))

    if matches:
        pairs = list(zip(matches, image_descriptions))
        for match, description in reversed(pairs):
            insert_text = (
                f"{match.group(0)}\n\n** ✅ Qwen Description:**\n{description}\n"
            )
            markdown_content = (
                markdown_content[: match.start()]
                + insert_text
                + markdown_content[match.end() :]
            )
        return markdown_content

    section_lines = ["\n\n---\n\n## 🖼️ Image Descriptions\n"]
    for i, desc in enumerate(image_descriptions, start=1):
        section_lines.append(f"\n### Image {i}\n\n** ✅ Qwen Description:**\n{desc}\n")

    return markdown_content + "".join(section_lines)
# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Build hierarchical JSON document structure
# ══════════════════════════════════════════════════════════════════════════════

@tool
def build_structure_tool(enhanced_markdown: str) -> dict:
    """
    Parse the enhanced markdown (after Qwen descriptions are spliced in) and
    build a hierarchical JSON structure keyed by page → headings (h1/h2/h3)
    → content + image_descriptions.

    Returns the full structure dict.
    """
    structure: dict = {"document": {"total_pages": 0, "pages": []}}

    # Split into pages using LlamaParse page-break markers
    page_blocks = re.split(r"(?:\n-{3,}\n|\f)", enhanced_markdown)
    if len(page_blocks) <= 1:
        page_blocks = [enhanced_markdown]

    for page_idx, block in enumerate(page_blocks):
        if not block.strip():
            continue

        page_num   = page_idx + 1
        page_entry = {"page": page_num, "headings": []}
        lines      = block.split("\n")

        current_h1: dict | None = None
        current_h2: dict | None = None
        pending_content_lines: list[str] = []
        inside_qwen_block = False
        pending_qwen_lines: list[str] = []

        def active_node() -> dict | None:
            if current_h2 is not None:
                return current_h2
            if current_h1 is not None:
                return current_h1
            return None

        def flush_content(target: dict) -> None:
            text = "\n".join(pending_content_lines).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text:
                existing = target.get("content", "")
                target["content"] = (existing + "\n" + text).strip() if existing else text
            pending_content_lines.clear()

        def flush_qwen(target: dict) -> None:
            text = "\n".join(pending_qwen_lines).strip()
            if text:
                target.setdefault("image_descriptions", []).append(text)
            pending_qwen_lines.clear()

        for line in lines:
            stripped = line.strip()

            # ── Qwen block start ──
            if re.match(r"^\*\*\s*✅\s*Qwen Description:\*\*", stripped):
                node = active_node()
                if node is not None:
                    flush_content(node)
                inside_qwen_block = True
                pending_qwen_lines.clear()
                continue

            # ── Inside Qwen block ──
            if inside_qwen_block:
                is_heading  = re.match(r"^#{1,3} ", stripped)
                is_new_qwen = re.match(r"^\*\*\s*✅\s*Qwen Description:\*\*", stripped)
                if is_heading or is_new_qwen:
                    node = active_node()
                    if node is not None:
                        flush_qwen(node)
                    inside_qwen_block = False
                    # fall through to heading handling
                else:
                    if stripped:
                        pending_qwen_lines.append(stripped)
                    continue

            # ── H1 ──
            if re.match(r"^# [^#]", stripped):
                node = active_node()
                if node:
                    flush_content(node)
                    flush_qwen(node)
                pending_content_lines.clear()
                title      = stripped.lstrip("#").strip()
                current_h1 = {
                    "title": title, "level": "h1",
                    "content": "", "image_descriptions": [], "subheadings": [],
                }
                current_h2 = None
                page_entry["headings"].append(current_h1)

            # ── H2 ──
            elif re.match(r"^## [^#]", stripped):
                node = active_node()
                if node:
                    flush_content(node)
                    flush_qwen(node)
                pending_content_lines.clear()
                title      = stripped.lstrip("#").strip()
                current_h2 = {
                    "title": title, "level": "h2",
                    "content": "", "image_descriptions": [], "subheadings": [],
                }
                if current_h1:
                    current_h1["subheadings"].append(current_h2)
                else:
                    page_entry["headings"].append(current_h2)

            # ── H3 ──
            elif re.match(r"^### [^#]", stripped):
                node = active_node()
                if node:
                    flush_content(node)
                    flush_qwen(node)
                pending_content_lines.clear()
                title = stripped.lstrip("#").strip()
                h3    = {
                    "title": title, "level": "h3",
                    "content": "", "image_descriptions": [],
                }
                if current_h2:
                    current_h2["subheadings"].append(h3)
                elif current_h1:
                    current_h1["subheadings"].append(h3)
                else:
                    page_entry["headings"].append(h3)
                current_h2 = h3  # h3 becomes active content target

            # ── Regular content ──
            else:
                if stripped:
                    pending_content_lines.append(stripped)

        # Flush end of page
        node = active_node()
        if node:
            flush_content(node)
            if inside_qwen_block:
                flush_qwen(node)

        # Clean empty fields
        def clean_node(n: dict) -> None:
            if n.get("content") == "":
                del n["content"]
            if n.get("image_descriptions") == []:
                del n["image_descriptions"]
            for sub in n.get("subheadings", []):
                clean_node(sub)
            if n.get("subheadings") == []:
                del n["subheadings"]

        for h in page_entry["headings"]:
            clean_node(h)

        if page_entry["headings"]:
            structure["document"]["pages"].append(page_entry)

    structure["document"]["total_pages"] = len(structure["document"]["pages"])
    return structure


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Extract protocol rules from markdown
# ══════════════════════════════════════════════════════════════════════════════

@tool
def extract_rules_tool(markdown_content: str) -> str:
    """
    Extract protocol/timing rule sentences from the markdown text.
    Returns a newline-joined string of up to 50 matching lines.
    """
    keywords = ["must", "must not", "only after", "before", "after", "wait", "assert"]
    rules    = [
        line.strip()
        for line in markdown_content.split("\n")
        if any(k in line.lower() for k in keywords)
    ]
    return "\n".join(rules[:50])


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers (not exposed as LangChain tools)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_all_images_from_pdf(pdf_path: str) -> list[dict]:
    """Fallback: enumerate every image in the PDF via PyMuPDF."""
    doc             = fitz.open(pdf_path)
    image_metadata  = []
    image_counter   = 0

    for page_num in range(len(doc)):
        page       = doc[page_num]
        image_list = page.get_images(full=True)

        for img in image_list:
            xref  = img[0]
            pix   = fitz.Pixmap(doc, xref)
            rects = page.get_image_rects(xref)
            bbox  = list(rects[0]) if rects else [0, 0, pix.width, pix.height]

            image_metadata.append({
                "id":          image_counter,
                "page_number": page_num + 1,
                "bbox":        bbox,
                "xref":        xref,
                "width":       pix.width,
                "height":      pix.height,
            })
            image_counter += 1

    doc.close()
    return image_metadata


def _extract_image_from_pdf(
    pdf_path: str,
    image_info: dict,
    output_dir: str = None,
) -> str:
    """Extract one image from pdf_path using xref or bbox crop."""
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    doc         = fitz.open(pdf_path)
    page_number = image_info.get("page_number", 0)
    bbox        = image_info.get("bbox", [0, 0, 100, 100])
    image_id    = image_info.get("id", "unknown")
    xref        = image_info.get("xref", None)

    output_path = os.path.join(output_dir, f"image_{image_id}_page_{page_number}.png")

    # Prefer xref-based extraction (lossless)
    if xref is not None:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(output_path)
            doc.close()
            return output_path
        except Exception:
            pass

    # Fallback: crop the page at bbox
    page = doc[page_number - 1] if page_number > 0 else doc[0]
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    pix  = page.get_pixmap(matrix=fitz.Matrix(6, 6), clip=rect)
    pix.save(output_path)
    doc.close()
    return output_path


def get_context_window(markdown: str, pos: int, window: int = 500) -> str:
    """Return a ±window-char snippet of markdown centred on pos."""
    start = max(0, pos - window)
    end   = min(len(markdown), pos + window)
    return markdown[start:end]