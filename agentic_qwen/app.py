#!/usr/bin/env python3
"""
app.py — Streamlit UI for the LangGraph PDF → Qwen Vision agentic workflow.

The UI is intentionally thin: it collects credentials + the PDF upload, then
delegates ALL processing logic to the LangGraph agent defined in agents.py.
Result rendering (markdown viewer, document tree, JSON, downloads) is handled
here after the agent returns its final WorkflowState.
"""

import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LlamaParse + Qwen Vision (Agentic)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers (unchanged visual layer from original app)
# ══════════════════════════════════════════════════════════════════════════════

def display_document_structure_ui(structure: dict) -> None:
    pages = structure.get("document", {}).get("pages", [])
    total = structure.get("document", {}).get("total_pages", 0)

    st.markdown(f"**Total pages with content:** {total}")
    tabs = st.tabs(["🌲 Visual Tree", "📋 Raw JSON"])

    with tabs[0]:
        if not pages:
            st.info("No structured content found.")
            return
        for page in pages:
            with st.expander(f"📄 Page {page['page']}", expanded=(page["page"] == 1)):
                headings = page.get("headings", [])
                if not headings:
                    st.caption("No headings found on this page.")
                    continue
                for h in headings:
                    _render_heading(h, depth=0)

    with tabs[1]:
        st.json(structure)


def _render_heading(node: dict, depth: int) -> None:
    level              = node.get("level", "h1")
    title              = node.get("title", "")
    content            = node.get("content", "")
    image_descriptions = node.get("image_descriptions", [])
    subheadings        = node.get("subheadings", [])

    indent = depth * 20

    if level == "h1":
        icon, title_style = "🔷", "font-size:16px; font-weight:600; color:#1a1a2e;"
        border = "border-left: 4px solid #4361ee; padding-left:10px; margin-bottom:6px;"
    elif level == "h2":
        icon, title_style = "🔹", "font-size:14px; font-weight:500; color:#16213e;"
        border = "border-left: 3px solid #7209b7; padding-left:8px; margin-bottom:4px;"
    else:
        icon, title_style = "▪️", "font-size:13px; font-weight:500; color:#0f3460;"
        border = "border-left: 2px solid #a8dadc; padding-left:6px; margin-bottom:4px;"

    st.markdown(
        f'<div style="margin-left:{indent}px; {border}">'
        f'<span style="{title_style}">{icon} {title}</span></div>',
        unsafe_allow_html=True,
    )

    if content:
        preview = content[:200] + ("..." if len(content) > 200 else "")
        st.markdown(
            f'<div style="margin-left:{indent+16}px; margin-bottom:4px;">'
            f'<span style="font-size:12px; color:#555; font-style:italic;">{preview}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    for i, desc in enumerate(image_descriptions):
        preview = desc[:300] + ("..." if len(desc) > 300 else "")
        st.markdown(
            f'<div style="margin-left:{indent+16}px; margin-bottom:6px; '
            f"background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:6px 10px;\">"
            f'<span style="font-size:11px; font-weight:600; color:#166534;">🖼️ Image Description {i+1}:</span><br>'
            f'<span style="font-size:12px; color:#14532d;">{preview}</span></div>',
            unsafe_allow_html=True,
        )

    for sub in subheadings:
        _render_heading(sub, depth=depth + 1)


# ══════════════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("🤖 LlamaParse + Qwen Vision — Agentic Workflow")
    st.markdown(
        "PDF documents are enriched by a **LangGraph multi-agent pipeline**: "
        "parsing → rule extraction → image description → markdown splicing → structure building."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key_llamaparse = st.text_input(
            "LlamaParse API Key",
            value=os.environ.get("LLAMA_PARSE_API_KEY", ""),
            type="password",
        )
        api_key_openrouter = st.text_input(
            "OpenRouter API Key",
            value=os.environ.get("OPEN_ROUTER_API_KEY", ""),
            type="password",
        )

        st.divider()
        st.markdown("### 🤖 Agent Pipeline")
        st.markdown("""
        ```
        START
          │
          ▼
        parse_pdf_node        → LlamaParse
          │
          ▼
        extract_rules_node    → protocol rules
          │
          ▼
        process_images_node   → extract + Qwen Vision (loop)
          │
          ▼
        splice_markdown_node  → enrich markdown
          │
          ▼
        build_structure_node  → hierarchical JSON
          │
          ▼
        END
        ```
        """)

        st.divider()
        st.markdown("### 🔧 Technologies")
        st.markdown("""
        - **LangGraph**: agent orchestration
        - **LlamaParse**: AI PDF → markdown
        - **Qwen Vision**: multimodal image analysis
        - **PyMuPDF**: image extraction
        - **Streamlit**: web interface
        """)

    # ── Main area ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload & Run Agent")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="The agent pipeline will process this document end-to-end.",
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                pdf_path = tmp_file.name

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            if st.button("🚀 Run Agentic Workflow", type="primary", use_container_width=True):
                if not api_key_llamaparse:
                    st.error("❌ Please provide your LlamaParse API Key.")
                    return
                if not api_key_openrouter:
                    st.error("❌ Please provide your OpenRouter API Key.")
                    return

                # Inject keys into env so tools.py / parser.py pick them up
                os.environ["LLAMA_PARSE_API_KEY"] = api_key_llamaparse
                os.environ["OPEN_ROUTER_API_KEY"]  = api_key_openrouter

                # ── Import here so env vars are already set ──────────────────
                from agents import run_workflow

                # ── Status container ─────────────────────────────────────────
                status_box = st.status(
                    "🤖 Agent is running…",
                    expanded=True,
                    state="running",
                )

                try:
                    with status_box:
                        st.write("**Step 1/5** — Parsing PDF with LlamaParse …")
                        st.write("**Step 2/5** — Extracting protocol rules …")
                        st.write("**Step 3/5** — Extracting & describing images with Qwen Vision …")
                        st.write("**Step 4/5** — Splicing descriptions into markdown …")
                        st.write("**Step 5/5** — Building document hierarchy …")

                    # ── Run the LangGraph agent ───────────────────────────────
                    output_dir = tempfile.mkdtemp()
                    final_state = run_workflow(pdf_path, output_dir=output_dir)

                    status_box.update(label="✅ Agent finished!", state="complete", expanded=False)

                    # ── Surface any errors ────────────────────────────────────
                    errors = final_state.get("errors", [])
                    if errors:
                        with st.expander("⚠️ Agent errors (non-fatal)", expanded=False):
                            for err in errors:
                                st.warning(err)

                    # ── Show parsed markdown ─────────────────────────────────
                    with st.expander("📝 Parsed Markdown (raw)", expanded=False):
                        st.markdown(final_state.get("markdown_content", ""))

                    # ── Show document structure ───────────────────────────────
                    structure = final_state.get("document_structure", {})
                    if structure:
                        st.subheader("🗂️ Document Structure")
                        with st.expander(
                            "Page / Heading / Subheading / Content / Image Descriptions",
                            expanded=True,
                        ):
                            display_document_structure_ui(structure)

                        structure_json_str = json.dumps(structure, indent=2)
                        st.download_button(
                            label="📥 Download Structure JSON",
                            data=structure_json_str,
                            file_name=f"{Path(uploaded_file.name).stem}_structure.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                    # ── Show enhanced markdown ────────────────────────────────
                    enhanced = final_state.get("enhanced_markdown", "")
                    if enhanced:
                        with st.expander("📄 Enhanced Markdown (with image descriptions)", expanded=True):
                            st.markdown(enhanced)

                        st.download_button(
                            label="📥 Download Enhanced Markdown",
                            data=enhanced,
                            file_name=f"{Path(uploaded_file.name).stem}_enhanced.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                except Exception as exc:
                    status_box.update(label="❌ Agent failed", state="error", expanded=True)
                    st.error(f"❌ Agent error: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

    with col2:
        st.header("ℹ️ Agent Nodes")
        st.markdown("""
        | Node | Responsibility |
        |------|---------------|
        | `parse_pdf` | LlamaParse → markdown + image metadata |
        | `extract_rules` | Protocol/timing rules from text |
        | `process_images` | Extract images → Qwen Vision loop |
        | `splice_markdown` | Inject descriptions at Figure refs |
        | `build_structure` | Page → H1 → H2 → H3 JSON tree |
        """)

        st.markdown("### 🔗 Links")
        st.markdown("[LlamaParse](https://www.llamaindex.ai/llamaparse)")
        st.markdown("[OpenRouter](https://openrouter.ai)")
        st.markdown("[Qwen Models](https://huggingface.co/Qwen)")
        st.markdown("[LangGraph](https://langchain-ai.github.io/langgraph/)")


if __name__ == "__main__":
    main()
