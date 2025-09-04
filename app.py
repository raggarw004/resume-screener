# app.py
# AI-Powered Resume Screener (Embeddings + NLP)
# - Supports PDF, DOCX, TXT
# - Robust to Gradio upload types (filepath, binary dict, file-like)
# - Optional OCR fallback for scanned PDFs (tesseract + poppler)
# - Final score = 0.7 * semantic similarity + 0.3 * keyword coverage

import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import docx  # python-docx

# ---------- Optional OCR fallback ----------
def optional_ocr_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception:
        return ""
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
        return "\n".join(pytesseract.image_to_string(im) for im in images).strip()
    except Exception:
        return ""

# ---------- File reading utilities ----------
def _to_bytes_and_name(obj):
    """Normalize Gradio upload into (bytes, name)."""
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        return p.read_bytes(), p.name
    if hasattr(obj, "read"):
        try:
            data = obj.read()
        finally:
            try: obj.seek(0)
            except Exception: pass
        name = getattr(obj, "name", "upload")
        return data, Path(name).name
    if isinstance(obj, dict):
        data = obj.get("data", b"")
        name = obj.get("name", "upload")
        return data, Path(name).name
    return b"", "upload"

def read_pdf_bytes(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text
    except Exception:
        pass
    return optional_ocr_from_pdf_bytes(file_bytes)

def read_docx_bytes(file_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""

def read_txt_bytes(file_bytes: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(enc, errors="ignore").strip()
        except Exception:
            continue
    return ""

def extract_text_from_upload(upload) -> Tuple[str, str]:
    """Return (display_name, text)."""
    data, name = _to_bytes_and_name(upload)
    low = name.lower()
    if low.endswith(".pdf"):
        text = read_pdf_bytes(data)
    elif low.endswith(".docx"):
        text = read_docx_bytes(data)
    elif low.endswith(".txt"):
        text = read_txt_bytes(data)
    else:
        text = read_txt_bytes(data)  # best effort
    return name, (text or "")

# ---------- Scoring ----------
def build_embedder(model_name: str):
    return SentenceTransformer(model_name)

def compute_keyword_coverage(job_desc: str, resume_text: str) -> float:
    import re
    tok = lambda s: set(w for w in re.findall(r"[A-Za-z0-9_#+-]{2,}", s.lower()))
    jd_terms = tok(job_desc)
    if not jd_terms:
        return 0.0
    res_terms = tok(resume_text)
    return len(jd_terms & res_terms) / max(1, len(jd_terms))

def score_resumes(job_desc: str,
                  resumes: List[Tuple[str, str]],
                  model_name: str,
                  semantic_weight: float = 0.7,
                  keyword_weight: float = 0.3) -> pd.DataFrame:
    model = build_embedder(model_name)
    jd_emb = model.encode([job_desc], convert_to_tensor=True, normalize_embeddings=True)[0]

    rows = []
    for name, text in resumes:
        emb = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
        cos = util.cos_sim(jd_emb, emb).item()          # [-1,1]
        sem = (cos + 1.0) / 2.0                         # [0,1]
        kw = compute_keyword_coverage(job_desc, text)   # [0,1]
        final = semantic_weight * sem + keyword_weight * kw
        rows.append((name, round(sem, 4), round(kw, 4), round(final, 4)))

    df = pd.DataFrame(rows, columns=["candidate", "semantic_sim", "keyword_coverage", "final_score"])
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)

# ---------- UI ----------
MODEL_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]

INTRO_MD = """
### 🧠 AI-Powered Resume Screener (Embeddings + NLP)

1) Paste a Job Description  
2) Upload resumes (PDF/DOCX/TXT)  
3) Click **Rank Candidates**  

*Final Score = 0.7·Semantic + 0.3·Keyword (adjustable).  
For scanned PDFs, enable OCR (install tesseract + poppler).*
"""

def rank_handler(jd_text, uploads, model_name, sem_w, kw_w):
    if not jd_text or not jd_text.strip():
        raise gr.Error("Please paste a Job Description.")

    # Normalize uploads
    if uploads is None: uploads = []
    if not isinstance(uploads, list): uploads = [uploads]

    resumes, unreadable = [], []
    for up in uploads:
        name, text = extract_text_from_upload(up)
        if text:
            resumes.append((name, text))
        else:
            unreadable.append(name)

    if not resumes:
        raise gr.Error("No readable resumes. Try DOCX/TXT or a text-based PDF. "
                       "For scanned PDFs, install OCR (tesseract + poppler).")

    df = score_resumes(jd_text, resumes, model_name, sem_w, kw_w)

    out = Path("artifacts"); out.mkdir(exist_ok=True)
    csv_path = out / "ranking.csv"
    df.to_csv(csv_path, index=False)

    note = f"Skipped (unreadable): {', '.join(unreadable)}" if unreadable else ""
    return df, str(csv_path), note

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MD)

    jd = gr.Textbox(label="Job Description", lines=10, placeholder="Paste the JD here")

    files = gr.File(label="Upload resumes (PDF/DOCX/TXT)",
                    file_count="multiple",
                    type="binary",                 # works with dict{name,data}
                    file_types=[".pdf", ".docx", ".txt"])

    with gr.Row():
        model_dd = gr.Dropdown(MODEL_OPTIONS, value="sentence-transformers/all-MiniLM-L6-v2",
                               label="Embedding model")
        sem_w = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Semantic weight")
        kw_w = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Keyword weight")

    run_btn = gr.Button("🔎 Rank Candidates", variant="primary")

    table = gr.Dataframe(interactive=False, wrap=True,
                         label="Results (higher Final Score = better match)")
    csv_out = gr.File(label="Download CSV", interactive=False)
    msg = gr.Markdown()

    run_btn.click(rank_handler,
                  inputs=[jd, files, model_dd, sem_w, kw_w],
                  outputs=[table, csv_out, msg])

if __name__ == "__main__":
    import gradio as gr
    demo.launch(
        server_name="127.0.0.1",   # force localhost
        server_port=7860,          # predictable port
        inbrowser=True,            # auto-open your default browser
        share=False,               # set True if you want a public link
        show_error=True            # surface errors in the UI
    )
