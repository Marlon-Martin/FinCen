import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import fitz  # PyMuPDF
import csv

# ---- CONFIG: set your folder here ----
PDF_FOLDER = r"C:\Users\marlo\OneDrive\Desktop\Coding\FinCen\fincen_advisory_pdfs"  # <-- change me
OUT_CSV = "fincen_fraud_mapping.csv"
OUT_JSON = "fincen_fraud_mapping_details.json"

# ---- Keyword dictionaries: tweak freely ----
# Each fraud type has a list of keywords or regex patterns (case-insensitive).
FRAUD_KEYWORDS: Dict[str, List[str]] = {
    "money_laundering": [
        r"\bmoney laundering\b",
        r"\bplacement\b",
        r"\blayering\b",
        r"\bintegration\b",
        r"\bAML\b",
        r"\bBSA\b",
        r"\bSARs?\b",
        r"\bbeneficial owner(ship)?\b",
        r"\bshell compan(y|ies)\b",
    ],
    "structuring_smurfing": [
        r"\bstructuring\b",
        r"\bsmurf(ing|ers?)\b",
        r"\bCTR(s)?\b",
        r"\b\$?9,?000\b",
        r"\bund(er|)-reporting\b",
    ],
    "terrorist_financing": [
        r"\bterror(ist|ism) financing\b",
        r"\bTF\b",
        r"\bOFAC\b",
        r"\bsanctions?\b",
        r"\bdesignated persons?\b",
    ],
    "human_trafficking": [
        r"\bhuman trafficking\b",
        r"\bHT\b",
        r"\bsex trafficking\b",
        r"\bforced labor\b",
    ],
    "ransomware": [
        r"\bransomware\b",
        r"\bdouble extortion\b",
        r"\bRaaS\b",
        r"\bransom\b",
    ],
    "bec_business_email_compromise": [
        r"\bB(\.| )?E(\.| )?C\b",
        r"\bbusiness email compromise\b",
        r"\bfraudulent wire\b",
        r"\bspoof(ed|ing)\b",
        r"\bimpersonation\b",
    ],
    "check_fraud": [
        r"\bcheck fraud\b",
        r"\baltered checks?\b",
        r"\bforged( |-)checks?\b",
        r"\bwash(ed|ing) checks?\b",
    ],
    "romance_scam_elder_fraud": [
        r"\bromance scam(s)?\b",
        r"\belder(?:ly)? fraud\b",
        r"\bsweetheart scam\b",
    ],
    "synthetic_identity": [
        r"\bsynthetic identit(y|ies)\b",
        r"\bC P N\b",
        r"\bSSN (mismatch|mismatched)\b",
    ],
    "crypto_virtual_assets": [
        r"\bvirtual asset(s)?\b",
        r"\bVASP(s)?\b",
        r"\bexchange wallet\b",
        r"\bmixer(s)?\b",
        r"\btumbler(s)?\b",
        r"\bblockchain\b",
        r"\bDeFi\b",
        r"\bstablecoin(s)?\b",
        r"\bcrypto(currency)?\b",
    ],
    "trade_based_money_laundering": [
        r"\bTBML\b",
        r"\btrade-?based money laundering\b",
        r"\bover-?invoicing\b",
        r"\bunder-?invoicing\b",
    ],
    "sanctions_evasion": [
        r"\bsanctions evasion\b",
        r"\bcircumvention\b",
        r"\bfront\b",
        r"\bthird-?country routing\b",
    ],
    "corruption_bribery": [
        r"\bFCPA\b",
        r"\bbriber(y|ies)\b",
        r"\bcorruption\b",
        r"\bkickbacks?\b",
        r"\bPEPs?\b",
    ],
    "mortgage_financial_aid_fraud": [
        r"\bmortgage fraud\b",
        r"\bPPP\b",
        r"\bEIDL\b",
        r"\bCARES Act\b",
        r"\bloan (forgiveness|forgiven)\b",
    ],
    "healthcare_fraud": [
        r"\bhealth care fraud\b",
        r"\bmedical billing\b",
        r"\bupcoding\b",
        r"\bunbundling\b",
    ],
    "tax_evasion": [
        r"\btax evasion\b",
        r"\bfalse returns?\b",
        r"\bundisclosed income\b",
    ],
}

# ---------------- Utilities ----------------


def is_image_based_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
    doc = fitz.open(pdf_path)
    n = len(doc)
    check = min(sample_pages, n)
    text_pages = 0
    for i in range(check):
        page = doc[i]
        txt = page.get_text()
        if txt and txt.strip():
            text_pages += 1
    doc.close()
    return text_pages == 0


def pdf_to_images(pdf_path: Path, out_dir: Path, scale: float = 2.5) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_paths = []

    # try a local import of Pillow; if unavailable, fall back to PyMuPDF's pix.save
    try:
        from PIL import Image as PILImage  # local import to avoid top-level import error
        use_pil = True
    except Exception:
        PILImage = None
        use_pil = False

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        p = out_dir / f"{pdf_path.stem}_page_{i+1:04d}.png"

        if use_pil and PILImage is not None:
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(p)
        else:
            # fallback: save directly with PyMuPDF (supports PNG)
            pix.save(str(p))

        img_paths.append(p)
    doc.close()
    return img_paths

def ocr_images_to_text(
    image_paths: List[Path], lang: str = "en"
) -> Tuple[str, List[dict]]:
    # Try PaddleOCR first, fall back to pytesseract+Pillow if PaddleOCR isn't available.
    try:
        import importlib

        # dynamically import paddleocr to avoid static analysis errors when the package
        # is not installed in the environment used by linters/IDEs
        mod = importlib.import_module("paddleocr")
        PaddleOCR = getattr(mod, "PaddleOCR", None)
        use_paddle = PaddleOCR is not None
    except Exception:
        PaddleOCR = None
        use_paddle = False

    all_text = []
    details = []

    if use_paddle:
        ocr = PaddleOCR(lang=lang, use_angle_cls=True)
        for p in image_paths:
            res = ocr.ocr(str(p), cls=True)
            page_items = []
            page_text_parts = []
            for line in res:
                for det in line:
                    text = det[1][0]
                    # confidence may be a string; coerce if possible
                    try:
                        conf = float(det[1][1])
                    except Exception:
                        conf = None
                    bbox = det[0]
                    page_items.append(
                        {"image": str(p), "text": text, "confidence": conf, "bbox": bbox}
                    )
                    page_text_parts.append(text)
            details.append({"image": str(p), "items": page_items})
            all_text.append("\n".join(page_text_parts))
    else:
        # Fallback: use pytesseract (requires Pillow and pytesseract installed)
        try:
            from PIL import Image as PILImage
            import pytesseract  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OCR backend not available: install 'paddleocr' (and paddlepaddle) or "
                "'pytesseract' with Pillow. Example:\n"
                "  pip install paddleocr\n"
                "  pip install paddlepaddle -f https://paddlepaddle.org.cn/whl/mkl/avx/stable.html\n"
                "or\n"
                "  pip install pytesseract pillow\n"
            ) from e

        for p in image_paths:
            img = PILImage.open(p)
            # pytesseract uses Tesseract language codes; 'eng' is used here for English
            try:
                text = pytesseract.image_to_string(img, lang="eng")
            except Exception:
                text = pytesseract.image_to_string(img)
            lines = [l for l in text.splitlines() if l.strip()]
            page_items = [
                {"image": str(p), "text": l, "confidence": None, "bbox": None}
                for l in lines
            ]
            details.append({"image": str(p), "items": page_items})
            all_text.append("\n".join(lines))

    return "\n\n".join(all_text), details


def extract_text_from_pdf(pdf_path: Path, tmp_dir: Path) -> Tuple[str, dict]:
    """
    Returns (full_text, meta) where meta contains:
      - mode: 'direct' or 'ocr'
      - pages: number of pages
      - ocr_details: list (only for OCR mode)
    """
    meta = {"file": str(pdf_path), "mode": None, "pages": 0, "ocr_details": None}
    doc = fitz.open(pdf_path)
    meta["pages"] = len(doc)
    # Try direct text first
    text_joined = []
    for i in range(len(doc)):
        page_txt = doc[i].get_text("text")
        text_joined.append(page_txt)
    doc.close()
    full = "\n".join(text_joined).strip()
    if full:
        meta["mode"] = "direct"
        return full, meta

    # Fallback to OCR (image-based)
    meta["mode"] = "ocr"
    images = pdf_to_images(pdf_path, tmp_dir / f"{pdf_path.stem}_pages")
    full_text, details = ocr_images_to_text(images, lang="en")
    meta["ocr_details"] = details
    return full_text, meta


def compile_patterns(mapping: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for label, pats in mapping.items():
        compiled[label] = [re.compile(p, flags=re.IGNORECASE) for p in pats]
    return compiled


def score_fraud_types(
    text: str, patterns: Dict[str, List[re.Pattern]]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Returns:
      - counts per label
      - matched snippets per label (unique, truncated)
    """
    counts = {label: 0 for label in patterns}
    matches = {label: [] for label in patterns}
    for label, regs in patterns.items():
        for rgx in regs:
            for m in rgx.finditer(text):
                counts[label] += 1
                snippet = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                matches[label].append(snippet)
    # Deduplicate snippets per label (keep order)
    for label in matches:
        seen = set()
        uniq = []
        for s in matches[label]:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        matches[label] = uniq[:20]  # cap to avoid giant JSON
    return counts, matches


# ---------------- Main pipeline ----------------


def main(pdf_dir: str, out_csv: str, out_json: str):
    base = Path(pdf_dir)
    assert base.exists(), f"Folder not found: {pdf_dir}"
    tmp_dir = base / "_ocr_tmp"
    tmp_dir.mkdir(exist_ok=True)

    compiled = compile_patterns(FRAUD_KEYWORDS)

    rows = []
    details_bundle = []

    pdfs = sorted([p for p in base.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        print("No PDFs found.")
        return

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Processing: {pdf.name}")
        try:
            text, meta = extract_text_from_pdf(pdf, tmp_dir)
            counts, matched_snips = score_fraud_types(text, compiled)

            total_hits = sum(counts.values())
            # Choose top labels with non-zero hits
            top = sorted(
                [(lbl, c) for lbl, c in counts.items() if c > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            top_labels = [f"{lbl}:{cnt}" for lbl, cnt in top]

            rows.append(
                {
                    "file": str(pdf),
                    "mode": meta["mode"],
                    "pages": meta["pages"],
                    "total_hits": total_hits,
                    "top_labels": "; ".join(top_labels),
                }
            )

            details_bundle.append(
                {
                    "file": str(pdf),
                    "meta": meta,
                    "counts": counts,
                    "top_labels": top,
                    "matched_snippets": matched_snips,
                }
            )

        except Exception as e:
            print(f"  !! Error on {pdf.name}: {e}")
            rows.append(
                {
                    "file": str(pdf),
                    "mode": "error",
                    "pages": None,
                    "total_hits": None,
                    "top_labels": f"ERROR: {e}",
                }
            )

    # write CSV using the standard library to avoid requiring pandas
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        # fallback header if no rows were produced
        fieldnames = ["file", "mode", "pages", "total_hits", "top_labels"]

    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure all keys present for DictWriter
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(details_bundle, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Summary CSV: {out_csv}")
    print(f"Details JSON: {out_json}")
    print("Tip: open the JSON to inspect which snippets matched for each fraud type.")


if __name__ == "__main__":
    # Allow quick override from CLI: python fincen_ocr_fraud_mapper.py "C:\folder" out.csv out.json
    if len(sys.argv) >= 2:
        PDF_FOLDER = sys.argv[1]
    if len(sys.argv) >= 3:
        OUT_CSV = sys.argv[2]
    if len(sys.argv) >= 4:
        OUT_JSON = sys.argv[3]
    main(PDF_FOLDER, OUT_CSV, OUT_JSON)
