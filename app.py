import gradio as gr
import json
import os
import re
import tempfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ── Fix DynamicCache seen_tokens compatibility ─────────────────
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(
            lambda self: sum(
                k.shape[-2] for k in self.key_cache if k is not None
            ) if hasattr(self, 'key_cache') and self.key_cache else 0
        )
except Exception:
    pass
# ── Config ─────────────────────────────────────────────────────
MODEL_REPO = os.environ.get("MODEL_REPO", "ratulsur/ap-auditor")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """You are a senior Accounts Payable Auditor AI.
Given one or more invoice records (as raw text or JSON), analyze them for financial risks.
Output ONLY a single valid JSON object:
{
  "risk_score": float (0.0 to 1.0),
  "risk_level": "low|medium|high|critical",
  "flags": [list of detected issues],
  "recommended_action": "approve|review|hold|reject",
  "explanation": "brief explanation",
  "savings_opportunity": float or null,
  "extracted_data": {
    "invoice_id": "string or null",
    "vendor": "string or null",
    "date": "YYYY-MM-DD or null",
    "currency": "string or null",
    "total_amount": float or null
  }
}
Possible flags: duplicate_invoice, price_mismatch, unapproved_vendor,
missing_po_reference, tax_discrepancy, round_number_fraud,
split_invoice, weekend_submission.
No explanation outside JSON."""

RISK_COLORS  = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
ACTION_ICONS = {"approve": "✅", "review": "🔍", "hold": "⏸️", "reject": "❌"}

# ── Model loader ───────────────────────────────────────────────
_pipe = None

def get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    print(f"Loading model: {MODEL_REPO}")
    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}")
    if use_gpu:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            quantization_config=bnb,
            device_map="cuda:0",
            token=HF_TOKEN or None,
            trust_remote_code=True,
            use_cache=False,           # ← fixes DynamicCache error
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            device_map="cpu",
            token=HF_TOKEN or None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,           # ← fixes DynamicCache error
        )
    tok = AutoTokenizer.from_pretrained(
        MODEL_REPO, token=HF_TOKEN or None, trust_remote_code=True,
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    model.eval()
    _pipe = pipeline(
        "text-generation", model=model,
        tokenizer=tok, return_full_text=False,
    )
    print("✓ Model ready on GPU" if use_gpu else "✓ Model ready on CPU")
    return _pipe
# ── Phi-3.5 prompt builder ─────────────────────────────────────
def build_prompt(content: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\nAudit this invoice record:\n\n{content}<|end|>\n"
        f"<|assistant|>\n"
    )

# ── Document extractors ────────────────────────────────────────
def extract_pdf(path: str) -> str:
    try:
        import pdfplumber
        chunks = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    chunks.append(f"[Page {i+1}]\n{t.strip()}")
                for table in page.extract_tables():
                    if table and table[0]:
                        try:
                            import pandas as pd
                            df = pd.DataFrame(table[1:], columns=table[0])
                            chunks.append(f"[Table]\n{df.to_string(index=False)}")
                        except Exception:
                            pass
        text = "\n\n".join(chunks)
        if len(text.strip()) < 50:
            return extract_ocr(path)
        return text
    except Exception:
        return extract_ocr(path)

def extract_ocr(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        if w < 1000:
            img = img.resize((1000, int(h * 1000 / w)))
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        return f"[OCR failed: {e}]"

def extract_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        if w < 1000:
            img = img.resize((1000, int(h * 1000 / w)))
        return pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip()
    except Exception as e:
        return f"[OCR failed: {e}]"

def extract_csv(path: str) -> str:
    import pandas as pd
    return pd.read_csv(path).to_string(index=False)

def extract_excel(path: str) -> str:
    import pandas as pd
    return pd.read_excel(path).to_string(index=False)

def extract_txt(path: str) -> str:
    with open(path, errors="ignore") as f:
        return f.read()

def extract_json_file(path: str) -> str:
    with open(path) as f:
        return json.dumps(json.load(f), indent=2)

EXTRACTORS = {
    ".pdf":  extract_pdf,
    ".png":  extract_image,
    ".jpg":  extract_image,
    ".jpeg": extract_image,
    ".tiff": extract_image,
    ".bmp":  extract_image,
    ".csv":  extract_csv,
    ".xlsx": extract_excel,
    ".xls":  extract_excel,
    ".txt":  extract_txt,
    ".json": extract_json_file,
}

def extract_document(file_obj) -> tuple:
    if file_obj is None:
        return "", ""
    path = file_obj.name
    ext  = Path(path).suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
        return "", f"Unsupported file type: {ext}"
    try:
        return extractor(path), ""
    except Exception as e:
        return "", str(e)

# ── Audit engine ───────────────────────────────────────────────
def run_audit(content: str) -> dict:
    p   = get_pipe()
    out = p(
        build_prompt(content),
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=p.tokenizer.eos_token_id,
        eos_token_id=p.tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    raw = re.sub(r"```json\s*|```\s*", "", out[0]["generated_text"]).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = (raw
                .replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
                .replace(",\n}", "\n}")
                .replace(",\n]", "\n]"))
        match = re.search(r"\{.*\}", raw2, re.DOTALL)
        try:
            return json.loads(match.group() if match else raw2)
        except Exception:
            return {"error": "Failed to parse model output", "raw": raw[:300]}

def build_summary(result: dict) -> str:
    risk_level  = result.get("risk_level", "unknown")
    risk_score  = result.get("risk_score", 0)
    action      = result.get("recommended_action", "review")
    flags       = result.get("flags", [])
    explanation = result.get("explanation", "")
    savings     = result.get("savings_opportunity")
    extracted   = result.get("extracted_data", {})

    bar   = "█" * int(risk_score * 20) + "░" * (20 - int(risk_score * 20))
    lines = [
        f"## {RISK_COLORS.get(risk_level, '⚪')} Risk Level: **{risk_level.upper()}**",
        f"**Score:** `[{bar}]` {risk_score:.0%}",
        f"**Action:** {ACTION_ICONS.get(action, '🔍')} **{action.upper()}**",
    ]
    if extracted:
        ext = []
        if extracted.get("vendor"):     ext.append(f"Vendor: {extracted['vendor']}")
        if extracted.get("invoice_id"): ext.append(f"Invoice ID: {extracted['invoice_id']}")
        if extracted.get("date"):       ext.append(f"Date: {extracted['date']}")
        if extracted.get("total_amount") is not None:
            ext.append(f"Total: {extracted.get('currency','INR')} {extracted['total_amount']:,.2f}")
        if ext:
            lines.append("\n**Extracted:**\n" + "\n".join(f"- {l}" for l in ext))
    if flags:
        lines.append("\n**Flags detected:**\n" + "\n".join(f"- `{f}`" for f in flags))
    else:
        lines.append("\n**Flags:** None ✓")
    if savings:
        lines.append(f"\n**Savings opportunity:** INR {savings:,.2f}")
    lines.append(f"\n**Explanation:** {explanation}")
    return "\n\n".join(lines)

# ── Main process ───────────────────────────────────────────────
def process(file_obj, json_text: str, raw_text: str):
    content = ""
    source  = ""

    if file_obj is not None:
        extracted, err = extract_document(file_obj)
        if err:
            return f"❌ File error: {err}", "", ""
        if not extracted.strip():
            return "❌ Could not extract text from file.", "", ""
        content = extracted
        source  = f"📄 `{Path(file_obj.name).name}`"
    elif json_text.strip():
        try:
            json.loads(json_text)
            content = json_text.strip()
            source  = "📋 JSON input"
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON: {e}", "", ""
    elif raw_text.strip():
        content = raw_text.strip()
        source  = "📝 Raw text"
    else:
        return "⚠️ Please upload a file, paste JSON, or enter raw text.", "", ""

    try:
        result = run_audit(content)
        if result.get("error"):
            return f"❌ {result['error']}", json.dumps(result, indent=2), ""
        summary = build_summary(result)
        preview = content[:2000] + ("…" if len(content) > 2000 else "")
        return (
            f"{source}\n\n{summary}",
            json.dumps(result, indent=2, ensure_ascii=False),
            preview,
        )
    except Exception as e:
        return f"❌ Error: {e}", "", ""

# ── Examples ───────────────────────────────────────────────────
EX_CLEAN = json.dumps({
    "invoice_id": "INV-88821", "vendor": "Infosys BPO Ltd.",
    "service": "Cloud Infrastructure Management", "date": "2024-11-15",
    "subtotal": 42500.0, "tax_rate_pct": 18.0, "tax_amount": 7650.0,
    "total_amount": 50150.0, "currency": "INR",
    "po_reference": "PO-4521", "payment_terms": "Net 30"
}, indent=2)

EX_DUPLICATE = json.dumps([
    {"invoice_id": "INV-11111", "vendor": "Wipro Technologies",
     "service": "SAP Integration Consulting", "date": "2024-10-01",
     "subtotal": 85000.0, "tax_rate_pct": 18.0, "tax_amount": 15300.0,
     "total_amount": 100300.0, "currency": "INR",
     "po_reference": "PO-7890", "payment_terms": "Net 30"},
    {"invoice_id": "INV-22222", "vendor": "Wipro Technologies",
     "service": "SAP Integration Consulting", "date": "2024-10-15",
     "subtotal": 85000.0, "tax_rate_pct": 18.0, "tax_amount": 15300.0,
     "total_amount": 100300.0, "currency": "INR",
     "po_reference": "PO-7890", "payment_terms": "Net 30"},
], indent=2)

EX_MULTI = json.dumps({
    "invoice_id": "INV-99999", "vendor": "Unknown Vendor Pvt. Ltd.",
    "service": "Consulting Services", "date": "2024-11-09",
    "subtotal": 100000.0, "tax_rate_pct": 28.0, "tax_amount": 28000.0,
    "total_amount": 128000.0, "currency": "INR",
    "po_reference": None, "payment_terms": "Net 15"
}, indent=2)

EX_SPLIT = json.dumps([
    {"invoice_id": "INV-33333", "vendor": "Deloitte Advisory",
     "service": "Risk Assessment", "date": "2024-12-01",
     "subtotal": 49500.0, "tax_rate_pct": 18.0, "tax_amount": 8910.0,
     "total_amount": 58410.0, "currency": "INR",
     "po_reference": "PO-8821", "payment_terms": "Net 30"},
    {"invoice_id": "INV-44444", "vendor": "Deloitte Advisory",
     "service": "Risk Assessment", "date": "2024-12-03",
     "subtotal": 48000.0, "tax_rate_pct": 18.0, "tax_amount": 8640.0,
     "total_amount": 56640.0, "currency": "INR",
     "po_reference": "PO-8821", "payment_terms": "Net 30"},
], indent=2)

EX_RAW = """INVOICE
Vendor:     TCS Consulting Ltd.
Invoice No: TCS-2024-5521
Date:       2024-11-09
Service:    Data Engineering Services   INR 49,000.00
GST @ 18%:                              INR  8,820.00
TOTAL DUE:                              INR 57,820.00
Payment Terms: Net 30
PO Reference: None"""

EX_SAP = """|DocNo     |Vendor              |Amount        |Curr|Status  |
|----------|--------------------|--------------|----|--------|
|1900045621|Wipro Limited       | 4,25,000.00  |INR |Open    |
|1900045622|HCL Technologies    | 2,10,500.00  |INR |Cleared |
|1900045623|Unknown Vendor      | 8,75,200.00  |INR |Open    |
Total: 15,10,700.00 INR"""

# ── UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="AP Auditor") as demo:
    gr.Markdown("""
# 🔍 Accounts Payable Auditor
**AI-powered invoice audit** — upload any document or paste invoice data.
Detects duplicates, fraud, pricing errors, tax issues, and compliance violations instantly.

*Fine-tuned Phi-3.5-mini-instruct · QLoRA 4-bit NF4 · r=64 · 1,219 training samples*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input — choose any one")
            file_in = gr.File(
                label="📎 Upload document (PDF, image, CSV, Excel, TXT, JSON)",
                file_types=[".pdf",".png",".jpg",".jpeg",".tiff",".bmp",
                            ".csv",".xlsx",".xls",".txt",".json"],
            )
            gr.Markdown("**— or paste invoice JSON —**")
            json_in = gr.Code(label="Invoice JSON", language="json", lines=8)
            gr.Markdown("**— or paste raw invoice text —**")
            text_in = gr.Textbox(
                label="Raw text (SAP export, email copy-paste, etc.)",
                lines=6, placeholder="Paste invoice text here...",
            )
            audit_btn = gr.Button("🔍 Audit Invoice", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Audit Result")
            summary_out = gr.Markdown()
            json_out    = gr.Code(label="Full JSON output", language="json", lines=16)

    with gr.Accordion("📄 Extracted text preview", open=False):
        preview_out = gr.Textbox(lines=5, interactive=False)

    gr.Markdown("### Try an example")
    with gr.Row():
        btn_clean = gr.Button("✅ Clean Invoice")
        btn_dup   = gr.Button("🔴 Duplicate")
        btn_multi = gr.Button("🟠 Multiple Flags")
        btn_split = gr.Button("⚠️ Split Invoice")
        btn_raw   = gr.Button("📝 Raw Text")
        btn_sap   = gr.Button("🗂️ SAP Export")

    gr.Markdown("""
---
**Supported formats:** PDF · Invoice image (PNG/JPG/TIFF) · CSV · Excel · JSON · Plain text · SAP ALV export

**Detects:** `duplicate_invoice` · `price_mismatch` · `unapproved_vendor` ·
`missing_po_reference` · `tax_discrepancy` · `round_number_fraud` ·
`split_invoice` · `weekend_submission`

**Model:** [ratulsur/ap-auditor](https://huggingface.co/ratulsur/ap-auditor) —
Fine-tuned Phi-3.5-mini-instruct · QLoRA · Apache 2.0
    """)

    audit_btn.click(
        fn=process,
        inputs=[file_in, json_in, text_in],
        outputs=[summary_out, json_out, preview_out],
    )
    btn_clean.click(fn=lambda: (None, EX_CLEAN,     ""), outputs=[file_in, json_in, text_in])
    btn_dup.click(  fn=lambda: (None, EX_DUPLICATE, ""), outputs=[file_in, json_in, text_in])
    btn_multi.click(fn=lambda: (None, EX_MULTI,     ""), outputs=[file_in, json_in, text_in])
    btn_split.click(fn=lambda: (None, EX_SPLIT,     ""), outputs=[file_in, json_in, text_in])
    btn_raw.click(  fn=lambda: (None, "",    EX_RAW),    outputs=[file_in, json_in, text_in])
    btn_sap.click(  fn=lambda: (None, "",    EX_SAP),    outputs=[file_in, json_in, text_in])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)