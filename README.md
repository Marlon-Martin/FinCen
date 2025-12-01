# **FinCEN Fraud Intelligence Platform**

### **Turning 30 years of FinCEN publications into a real-time fraud-analysis engine.**

**Authors:**
Tim Goncharov · Mahdi Bahreman · Natalia Herrera · Marlon Martin

---

# **Project Summary (First Impression)**

**A full-stack system that ingests every FinCEN Advisory/Alert/Notice since 1996, extracts text with OCR, generates structured LLM summaries, classifies fraud families, stores results in Supabase, and exposes fraud insights through a deployed Streamlit dashboard.**

> *Goal: Provide AML and fraud-risk teams with instant visibility into emerging trends, red flags, and SAR-relevant schemes.*

---

# **Quick Start (Setup & Running)**

## **Clone + Install**

```bash
uv sync
```

Or:

```bash
pip install -r requirements.txt
```

---

## **Environment Variables**

Create `.env`:

```bash
GEMINI_API_KEY="your_key"      
SUPABASE_URL="your_supabase_url"
SUPABASE_SERVICE_KEY="your_service_role_key" 
```

---

## **Run the Full Pipeline**

```bash
python fincen_publications_crawler.py
python fincen_text_extractor.py
python fincen_ocr_fraud_mapper.py
python fincen_summary_generator.py
python build_vectorstore.py
```

Everything populates directly into Supabase:

* publication metadata (`fincen_publications`)
* full-document text (`fincen_fulltext`)
* LLM summaries (`fincen_llm_summaries`)
* fraud family mappings + semantic chunks (`fincen_fraud_mapping`, `fincen_semantic_chunks`, `fincen_keyword_locations`)

**No CSVs are used in the live pipeline.**

---

## **Run the Dashboard**

```bash
uv run streamlit run streamlit_app.py
```

Or view it on Modal:

👉 **Live Streamlit App:**
**[https://tjgoncharov--fincen-fraud-analytics-serve-streamlit.modal.run/](https://tjgoncharov--fincen-fraud-analytics-serve-streamlit.modal.run/)**

---

# Visuals / Application Design

## **Architecture Diagram**

<img width="750" height="600" alt="architecture" src="https://github.com/user-attachments/assets/f45109b7-4d62-4647-b798-876e7cf1e8a6" />



---

## **UI Screenshots**

### **FinCEN Insights Tab**

*(Momentum of fraud families, narrative summaries, top red flags)*

![571555F2-7AC1-457A-B25A-499A4BD61B76_1_105_c](https://github.com/user-attachments/assets/8e5899c7-8775-4736-8562-3f6d7b48d28b)


### **Timeline View**

*(Counts fraud families per year 1996–2025)*

![5F54EAAF-BF20-46FA-AB3F-31B382A4C37E_1_105_c](https://github.com/user-attachments/assets/cea40e33-c37f-4a15-8ef3-95a33c7abcc7)


### **Semantic Search**

*(Embedding-powered search across all FinCEN text)*

![E4B3962F-0FF3-4FB5-86E3-D44B2960CE08_1_105_c](https://github.com/user-attachments/assets/282ccb8c-1775-4be6-8ae5-2a09404f8139)


---

## **What We Did (with code snippets)**

### **🔹 1. Unified Crawler (Publications → Supabase)**

Our new crawler replaces the old two-script system.  
It scrapes all Advisories, Alerts, and Notices → downloads their PDFs → uploads to Supabase Storage → inserts metadata into `fincen_publications`.

```python
records.append({
    "fincen_id": fincen_id,
    "title": title,
    "doc_type": doc_type,
    "pdf_url": pdf_url,
    "date": published_date
})
```

Stored in Supabase table: **`fincen_publications`**.

---

### **🔹 2. OCR + Full-Document Text Extraction**

Each PDF is processed page-by-page.  
If PyMuPDF extracts no text (common for scanned PDFs), we fall back to OCR.

```python
text = page.get_text("text")
if not text.strip():
    text = ocr_engine.ocr(image)
```

All pages are joined into one full-document string and stored in Supabase table:  
**`fincen_fulltext`**.

---

### **🔹 3. LLM Summarization Input **

For summaries, we:

1. We take the full document text from `fincen_fulltext`.  
2. Apply a **character cap (~20,000 chars)** approximately 15 page cap.  
3. Send that text to Gemini to generate **one structured summary per document**.

```python
# Current safeguard: limit characters sent to LLM
prompt_text = full_text[:20000]
```

Just a safety measure.

---

### **🔹 4. LLM Structured Summaries (Fraud Families · Schemes · Red Flags)**

Gemini is prompted to return a **strict JSON schema** containing:

- `primary_fraud_families` (main fraud family discussed)  
- `secondary_fraud_families` (fraud family mentioned but not the main type)
- `specific_schemes` (family → scheme_label → description)  
- `key_red_flags` (SAR-relevant indicators)

If the model cannot confidently map a fraud family, it uses **`"other"`**, which allows us to later inspect and potentially define **new emerging families**.

```json
{
  "primary_fraud_families": ["sanctions_evasion"],
  "secondary_fraud_families": ["terrorist_financing"],
  "specific_schemes": [
    {
      "fraud_family": "sanctions_evasion",
      "scheme_label": "unhosted_wallet_routing",
      "description": "Use of multiple unregistered wallets..."
    }
  ],
  "key_red_flags": ["rapid outward transfers"]
}
```

Stored in Supabase table: **`fincen_llm_summaries`**.

---

### **🔹 5. Semantic Fraud Mapping (OCR Text + Embeddings)**

The fraud mapper works at the **chunk level** to classify fraud families and support future highlight overlays.

- Extracts paragraph-like chunks from each PDF.  
- Embeds each chunk using SentenceTransformers.  
- Compares against a curated fraud-family vector inventory via cosine similarity.

```python
emb = model.encode(chunk_text)
score = cosine_similarity(emb, fraud_family_vectors)
```

Stored in Supabase tables:

- **`fincen_fraud_mapping`** — document-level fraud summary.  
- **`fincen_semantic_chunks`** — chunk text + fraud labels + similarity scores.  
- **`fincen_keyword_locations`** — high-signal chunks for potential PDF highlighting.

---


### **🔹 6. Embeddings + Local Vectorstore (Semantic Search)**

Chunk-level embeddings are stored in a local vectorstore, which powers the Streamlit **Semantic Search** tab.

```python
vec = model.encode(chunk_text)
np.save("vecstore/embeddings.npy", vectors)
```

Output:

```
vecstore/
  embeddings.npy
  metadata.jsonl
  settings.json
```

Used by: **`semantic_search.py`** and the Streamlit search UI.

---

### **🔹 8. Streamlit Dashboard**

Three main tabs:

✔ **FinCEN Insights** — narrative summaries, top fraud families, repeated red flags.  
✔ **Fraud Families Timeline** — per-year fraud-family counts, with drill-down by year/family.  
✔ **Semantic Search** — embedding-based retrieval across chunk text + filters by fraud label/document.

---

# 🔍  Findings (What We Learned)

### **Why This Matters for AML / Financial Crime**

FinCEN documents are long, dense, and inconsistent.
Our system:

✔ Helps analysts spot emerging fraud patterns faster
✔ Centralizes 30 years of guidance
✔ Surfaces SAR-ready red flags
✔ Tracks geopolitical fraud trends (ML, sanctions, TF, crypto)
✔ Enables instant semantic retrieval of intelligence

---

## **Top Fraud Families (Counts Across 30 Years)**

*Money laundering dominates the corpus.*
Terrorist finacing follows second.
Sanctions evasion correlates with geopolitical events.


<img width="1076" height="160" alt="Screenshot 2025-12-01 005118" src="https://github.com/user-attachments/assets/ab0acfee-70b7-4088-b8ec-092b71bb113b" />

---

## **Fraud Intensity Over Time**

Shows per-year spikes in:

* terrorist financing
* corruption
* human trafficking
* ransomware
* crypto/virtual asset scams

![9D0BF9B5-5718-4BDC-AE6E-AAB36625F50C_1_105_c](https://github.com/user-attachments/assets/c0692c34-2b83-488b-8c4c-19d52b23f39d)


---

## **Semantic Search Insights**

Analysts can ask questions like:

*“P2P mule typologies 2024”*
*“shell company Russian sanctions guidance”*
*“romance scam SAR indicators”*

…and instantly retrieve relevant FinCEN text.

---

# Folder Structure 

```bash
.
├── fincen_publications_crawler.py
├── fincen_text_extractor.py
├── fincen_ocr_fraud_mapper.py
├── fincen_summary_generator.py
├── fincen_build_fraud_dictionary.py
├── build_vectorstore.py
├── semantic_search.py
├── supabase_helpers.py
├── streamlit_app.py
├── modal_app.py
├── vecstore/
├── requirements.txt
├── pyproject.toml
└── README.md
```



