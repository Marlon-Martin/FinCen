# **FinCEN Fraud Intelligence Platform**

### **Turning 30 years of FinCEN data into a searchable fraud-analysis engine.**

**Authors:**

* Tim Goncharov
* Mahdi Bahreman 
* Natalia Herrera
* Marlon Martin

---

# ⭐ **Summary**

A full-stack pipeline that scrapes every FinCEN Advisory, Alert, and Notice (1996–2025), extracts text using OCR, summarizes fraud-dense chunks using LLMs, classifies fraud families, and visualizes emerging financial-crime trends in a deployed Streamlit dashboard.

---

# ⚡ **Quick Start Guide**

### **Install**

```bash
uv sync
```

(or, if using pip: `pip install -r requirements.txt`)

---

### **Run the Pipeline**

```bash
python advisory_crawler.py
python alerts_notices_crawler.py
python fincen_ocr_fraud_mapper.py
python fincen_summary_generator.py
python json_to_csv_summaries.py
```

---

### **Environment Variables**

Create a `.env` file (or `.env.example`) like:

```bash
OPENAI_API_KEY="your_key"
SUPABASE_URL="your_url"
SUPABASE_KEY="your_key"
```

*(LLM key optional )*

---

# 🖥️ **Visuals / Application Design**

## **Architecture Diagram**
*“_End-to-End FinCEN Fraud Intelligence Architecture_”*

*<img width="532" height="346" alt="image" src="https://github.com/user-attachments/assets/e2baea62-563b-4309-ad78-297a7c07505a" />*

**Flow:**
PDF → OCR → Chunking → LLM Summaries → Fraud Mapping (regex + embeddings) → Supabase → Streamlit Dashboard

---

## **Screenshots / UI**

![571555F2-7AC1-457A-B25A-499A4BD61B76](https://github.com/user-attachments/assets/53517ce9-b7dd-4378-9cf9-b73578a8a3a1)


> The FinCEN Insights tab shows a narrative summary of the most active fraud families, emerging schemes, and repeating red flags.

---

## **Demo Link**

👉 **Live Streamlit App (deployed on Modal):**
**https://tjgoncharov--fincen-fraud-analytics-serve-streamlit.modal.run/**

---

# What We Did 

Break down of the pipeline with **short explanations + visual snippets**, matching rubric expectations.

---

## **🔎 1. Crawling FinCEN PDFs**

Scripts:

* `advisory_crawler.py`
* `alerts_notices_crawler.py`

**Snippet:**

```python
records.append({
    "date": date_str,
    "title": title,
    "url": pdf_url
})
```

**Output sample (`fincen_advisories.csv`):**

| date       | title                       | url          |
| ---------- | --------------------------- | ------------ |
| 2024-03-15 | FinCEN Issues Advisory on X | https://…pdf |

---

##  2. OCR + Text Extraction

We used **PyMuPDF + OCR fallback** to extract layout-aware text from each PDF.

```python
text = page.get_text("text")
```

**Output (`raw_text`):**

```
Multiple P2P payments received from unrelated accounts ...
Funds transferred rapidly across jurisdictions ...
```

---

## 3. Semantic Chunking

Using **SentenceTransformers**, we created ~500-token chunks to avoid summarizing irrelevant pages.

**Aha moment:**
👉 *Chunking before summarization reduced LLM cost massively.*

```python
emb = model.encode(chunks)
```

---

## 4. LLM Structured Summaries

Each chunk is summarized into this schema:

```json
{
  "primary_fraud_family": "money_laundering",
  "secondary_fraud_families": ["smuggling"],
  "key_schemes": ["cross-border mule activity"],
  "key_red_flags": ["multiple P2P deposits then rapid withdrawals"],
  "sar_guidance": "Identify unhosted wallet transfers..."
}
```

Saved to `fincen_summaries.json`.

---

## 5. Fraud Label Mapping

We combined:

* Regex (high-precision tags)
* Embeddings (semantic concepts)

**Snippet:**

```python
if re.search(pattern, text, re.I):
    labels.append("sanctions_evasion")
```

---

## 6. Supabase Storage

Tables:

| Table           | Description        |
| --------------- | ------------------ |
| `documents`     | PDF metadata       |
| `chunks`        | OCR’d chunk text   |
| `embeddings`    | vector embeddings  |
| `summaries`     | LLM JSON summaries |
| `fraud_mapping` | final fraud labels |

---

## 📊 7. Visual Dashboard (Streamlit)

Tabs include:

* **FinCEN Insights**
* **Fraud Families Timeline**
* **Semantic Search**
* **Document Drill-Down**

![5F54EAAF-BF20-46FA-AB3F-31B382A4C37E](https://github.com/user-attachments/assets/0f19394a-6efb-4459-9ef2-051be20cb6d9)
![1F13C882-942A-4BC5-8EEC-73B7EC084EDB](https://github.com/user-attachments/assets/fe3306c7-8c3c-41fe-b4bd-6f24e854c547)


---

# 🔍 Findings 

## Why Analysts Care

FinCEN publishes huge PDFs that are slow for AML teams to digest.
Our tool provides:

✔ **Fast identification of active fraud families**
✔ **Emerging scheme detection**
✔ **Red-flag extraction for SAR writing**
✔ **Historical trend analysis (1996–2025)**
✔ **Semantic search to find relevant guidance instantly**

---

## Key Visual Findings 

### **Top Fraud Labels (Counts)**

![5E2C402F-F7C3-48E5-98D9-3A35A4A3A7DC](https://github.com/user-attachments/assets/6eb7a08a-4823-4d75-a34b-139cdc4913af)


**Observed:**

* Money laundering dominates the dataset (~5k signals).
* Terrorist financing & sanctions evasion have strong cyclical spikes.
* Cybercrime and crypto-related schemes rise sharply after 2018.

---

### **Fraud Intensity Over Time**

![9D0BF9B5-5718-4BDC-AE6E-AAB36625F50C](https://github.com/user-attachments/assets/a72a3ef9-d01c-4c99-8df4-e2ed87536284)

**Insights:**

* Sanctions evasion surges around geopolitical events (2014, 2022).
* Human trafficking becomes more prominently flagged post-2020.
* Crypto/virtual asset fraud spikes in 2018 and remains elevated.

---

### **Semantic Search Discovery**

![E4B3962F-0FF3-4FB5-86E3-D44B2960CE08](https://github.com/user-attachments/assets/62b10147-0405-4a07-9e42-6ef6ba510caf)


Analysts can query:

* *“money laundering”*
* *“unhosted wallets sanctions”*
* *“mule typologies 2024”*

And instantly retrieve the most relevant chunks.

---

# 📁 **Folder Structure**

```bash
.
├── advisory_crawler.py
├── alerts_notices_crawler.py
├── fincen_ocr_fraud_mapper.py
├── fincen_summary_generator.py
├── json_to_csv_summaries.py
├── data/
│   ├── fincen_advisories.csv
│   ├── fincen_alerts.csv
│   ├── fincen_keyword_locations.csv
│   ├── fincen_summary_summaries.json
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# 🧪 **Run the Dashboard Locally**

```bash
streamlit run streamlit_app.py
```

