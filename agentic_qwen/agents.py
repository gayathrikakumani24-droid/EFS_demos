"""
agents.py — LangGraph agentic workflow for PDF → Qwen Vision enrichment.

Graph topology (linear pipeline with a conditional image-loop):

  parse_pdf_node
       │
       ▼
  extract_rules_node
       │
       ▼
  process_images_node   ← iterates over every image (internal loop)
       │
       ▼
  splice_markdown_node
       │
       ▼
  build_structure_node
       │
       ▼
  [END]

The WorkflowState TypedDict is the single shared state object threaded
through every node — no globals, no side channels.
"""

from __future__ import annotations

import os
import tempfile
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from tools import (
    parse_pdf_tool,
    extract_image_tool,
    describe_image_tool,
    splice_descriptions_tool,
    build_structure_tool,
    extract_rules_tool,
    get_context_window,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared workflow state
# ══════════════════════════════════════════════════════════════════════════════

class WorkflowState(TypedDict):
    # Inputs
    pdf_path: str
    output_dir: str

    # Intermediate
    markdown_content:   str
    image_metadata:     list[dict]
    rules_text:         str
    image_descriptions: list[str]

    # Outputs
    enhanced_markdown:  str
    document_structure: dict

    # Internal progress tracking
    current_image_index: int
    total_images:        int

    # Error sink (any node can write here)
    errors: Annotated[list[str], operator.add]


# ══════════════════════════════════════════════════════════════════════════════
# Node implementations
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf_node(state: WorkflowState) -> dict:
    """
    Node 1 — Call the parse_pdf_tool.
    Populates markdown_content, image_metadata, and total_images.
    """
    print("🔄 [Agent] parse_pdf_node: parsing PDF with LlamaParse …")
    try:
        result = parse_pdf_tool.invoke(state["pdf_path"])
        print(
            f"   ✅ Parsed. "
            f"Markdown length: {len(result['markdown_content'])} chars | "
            f"Images found: {len(result['image_metadata'])}"
        )
        return {
            "markdown_content":    result["markdown_content"],
            "image_metadata":      result["image_metadata"],
            "total_images":        len(result["image_metadata"]),
            "current_image_index": 0,
            "image_descriptions":  [],
        }
    except Exception as exc:
        print(f"   ❌ parse_pdf_node error: {exc}")
        return {"errors": [f"parse_pdf_node: {exc}"]}


def extract_rules_node(state: WorkflowState) -> dict:
    """
    Node 2 — Extract protocol rules from the parsed markdown.
    """
    print("📜 [Agent] extract_rules_node: extracting protocol rules …")
    try:
        rules = extract_rules_tool.invoke(state["markdown_content"])
        print(f"   ✅ Extracted {len(rules.splitlines())} rule lines.")
        return {"rules_text": rules}
    except Exception as exc:
        print(f"   ❌ extract_rules_node error: {exc}")
        return {"rules_text": "", "errors": [f"extract_rules_node: {exc}"]}


def process_images_node(state: WorkflowState) -> dict:
    """
    Node 3 — Iterate over every image in image_metadata.
    For each image: extract → describe → append description.

    This node runs once and processes ALL images sequentially so that
    progress feedback is clean and the state transition is simple.
    """
    image_metadata    = state.get("image_metadata", [])
    markdown_content  = state["markdown_content"]
    rules_text        = state.get("rules_text", "")
    output_dir        = state.get("output_dir", tempfile.gettempdir())

    if not image_metadata:
        print("ℹ️  [Agent] process_images_node: no images found — skipping.")
        return {"image_descriptions": []}

    print(f"🖼️  [Agent] process_images_node: processing {len(image_metadata)} image(s) …")
    descriptions: list[str] = []

    for idx, img_info in enumerate(image_metadata):
        print(f"   [{idx + 1}/{len(image_metadata)}] extracting image …", end=" ")
        try:
            # Step A — extract image file
            image_path = extract_image_tool.invoke(input={
                "payload":{
                "pdf_path":   state["pdf_path"],
                "image_info": img_info,
                "output_dir": output_dir,
                }
            })

            # Step B — build context window from markdown
            approx_pos   = int(len(markdown_content) * (idx / max(1, len(image_metadata))))
            context_text = get_context_window(markdown_content, approx_pos)

            # Step C — describe with Qwen
            description = describe_image_tool.invoke(input={
                "payload":{
                    "image_path":   image_path,
                "context_text": context_text,
                "rules_text":   rules_text,
            }
            })

            descriptions.append(description)
            print(f"✅ described ({len(description)} chars)")

        except Exception as exc:
            err_msg = f"Image {idx}: {exc}"
            print(f"❌ {err_msg}")
            descriptions.append(f"[Description unavailable: {exc}]")

    return {
        "image_descriptions":  descriptions,
        "current_image_index": len(image_metadata),
    }


def splice_markdown_node(state: WorkflowState) -> dict:
    """
    Node 4 — Splice Qwen descriptions into the markdown.
    """
    print("🔗 [Agent] splice_markdown_node: splicing descriptions …")
    try:
        enhanced = splice_descriptions_tool.invoke(input={
            "markdown_content":   state["markdown_content"],
            "image_descriptions": state.get("image_descriptions", []),
        })
        print(f"   ✅ Enhanced markdown length: {len(enhanced)} chars.")
        return {"enhanced_markdown": enhanced}
    except Exception as exc:
        print(f"   ❌ splice_markdown_node error: {exc}")
        # Graceful degradation — use the original markdown
        return {
            "enhanced_markdown": state["markdown_content"],
            "errors": [f"splice_markdown_node: {exc}"],
        }


def build_structure_node(state: WorkflowState) -> dict:
    """
    Node 5 — Build the hierarchical JSON document structure.
    """
    print("🗂️  [Agent] build_structure_node: building document hierarchy …")
    try:
        structure = build_structure_tool.invoke(state["enhanced_markdown"])
        total_pages = structure.get("document", {}).get("total_pages", 0)
        print(f"   ✅ Structure built. Total pages: {total_pages}.")
        return {"document_structure": structure}
    except Exception as exc:
        print(f"   ❌ build_structure_node error: {exc}")
        return {
            "document_structure": {},
            "errors": [f"build_structure_node: {exc}"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_workflow() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke / stream.
    """
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("parse_pdf",       parse_pdf_node)
    graph.add_node("extract_rules",   extract_rules_node)
    graph.add_node("process_images",  process_images_node)
    graph.add_node("splice_markdown", splice_markdown_node)
    graph.add_node("build_structure", build_structure_node)

    # Linear edges
    graph.add_edge(START,            "parse_pdf")
    graph.add_edge("parse_pdf",      "extract_rules")
    graph.add_edge("extract_rules",  "process_images")
    graph.add_edge("process_images", "splice_markdown")
    graph.add_edge("splice_markdown","build_structure")
    graph.add_edge("build_structure", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_workflow(pdf_path: str, output_dir: str | None = None) -> WorkflowState:
    """
    Execute the full agentic pipeline and return the final WorkflowState.

    Args:
        pdf_path:   Absolute path to the PDF to process.
        output_dir: Directory for extracted image files.
                    Defaults to the system's temp directory.

    Returns:
        The final WorkflowState with all populated fields.
    """
    workflow = build_workflow()

    initial_state: WorkflowState = {
        "pdf_path":             pdf_path,
        "output_dir":           output_dir or tempfile.gettempdir(),
        "markdown_content":     "",
        "image_metadata":       [],
        "rules_text":           "",
        "image_descriptions":   [],
        "enhanced_markdown":    "",
        "document_structure":   {},
        "current_image_index":  0,
        "total_images":         0,
        "errors":               [],
    }

    final_state = workflow.invoke(initial_state)
    return final_state