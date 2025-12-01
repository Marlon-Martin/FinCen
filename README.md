# **FinCEN Fraud Intelligence Platform**

### **Turning 30 years of FinCEN publications into a real-time fraud-analysis engine.**

**Authors:**
Tim Goncharov · Mahdi Bahreman · Natalia Herrera · Marlon Martin

---

# **Project Summary (First Impression)**

**A full-stack system that ingests every FinCEN Advisory/Alert/Notice since 1996, extracts text with OCR, summarizes fraud-dense chunks using LLMs, classifies fraud families, stores results in Supabase, and exposes fraud insights through a deployed Streamlit dashboard.**

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
OPENAI_API_KEY="your_key"       # optional (for LLM summaries)
SUPABASE_URL="your_supabase_url"
SUPABASE_SERVICE_KEY="your_service_role_key"  # required
```

(We also provide `.env.example` — fill in and rename to `.env`.)

---

## **Run the Full Pipeline**

```bash
python fincen_publications_crawler.py
python fincen_text_extractor.py
python fincen_ocr_fraud_mapper.py
python fincen_summary_generator.py
python fincen_build_fraud_dictionary.py
python build_vectorstore.py
```

Everything populates directly into Supabase:

* documents
* pages
* OCR text
* LLM summaries
* fraud family mappings
* embeddings
* semantic vectorstore

**No CSVs are used in the live pipeline.**

---

## **Run the Dashboard**

```bash
streamlit run streamlit_app.py
```

Or view it on Modal:

👉 **Live Streamlit App:**
**[https://tjgoncharov--fincen-fraud-analytics-serve-streamlit.modal.run/](https://tjgoncharov--fincen-fraud-analytics-serve-streamlit.modal.run/)**

---

# Visuals / Application Design

## **Architecture Diagram**

<img width="532" height="346" alt="image" src="https://github.com/user-attachments/assets/c2ab3c64-dea0-4384-b6fb-9c932ec29424" />



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

### **🔹 1. Unified Crawler**

Replaces two old crawlers → now one job:

```python
records.append({
    "doc_id": doc_id,
    "title": title,
    "pub_type": pub_type,
    "pdf_url": pdf_url,
    "published": published_date
})
```

Stored in Supabase table: `documents`.

---

### **🔹 2. OCR + Page-Level Extraction**

```python
text = page.get_text("text")
if not text.strip():
    text = ocr_page_image(page)
```

Stored in `pages` table.

---

### **🔹 3. Chunking Strategy (Aha Moment)**

➡️ **Chunk first, summarize only fraud-dense chunks**
This lowered LLM usage by **70–85%**.

```python
chunks = semantic_chunk(page_text, max_tokens=400)
```

---

### **🔹 4. LLM Summaries**

Each chunk becomes a structured JSON summary:

```json
{
  "primary_fraud_family": "money_laundering",
  "secondary_fraud_families": ["sanctions_evasion"],
  "key_schemes": ["shell company layering", "cross-border mules"],
  "key_red_flags": ["large P2P transfers followed by crypto conversion"]
}
```

Stored in `summaries` table.

---

### **🔹 5. Fraud Dictionary (Regex + Semantic)**

```python
if re.search(r"unhosted wallet", text, re.I):
    labels.append("crypto_virtual_assets")
```

The team-built fraud dictionary contains:

* fraud families
* synonyms
* indicators
* semantic keywords

---

### **🔹 6. Embeddings + Vectorstore**

```python
vec = model.encode(chunk_text)
supabase.table("embeddings").insert(...)
```

Used for:

* semantic search tab
* “find similar fraud patterns”

---

### **🔹 7. Streamlit Dashboard**

Three main tabs:

✔ **FinCEN Insights** — narrative summaries
✔ **Fraud Families Timeline** — trend visualization
✔ **Semantic Search** — embedding similarity

---

# 🔍 Findings (What We Learned)

### **Why This Matters for AML / Financial Crime**

FinCEN documents are long, dense, and inconsistent.
Our system:

✔ Identifies emerging fraud patterns before humans see them
✔ Centralizes 30 years of guidance
✔ Surfaces SAR-ready red flags
✔ Tracks geopolitical fraud trends (ML, sanctions, TF, crypto)
✔ Enables instant semantic retrieval of intelligence

---

## **Top Fraud Families (Counts Across 30 Years)**

*Money laundering dominates the corpus.*
Crypto-related fraud spikes since 2018.
Sanctions evasion correlates with geopolitical events.

![5E2C402F-F7C3-48E5-98D9-3A35A4A3A7DC_4_5005_c](https://github.com/user-attachments/assets/87a1b21f-3a40-460b-8b01-f561aff5b757)


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


