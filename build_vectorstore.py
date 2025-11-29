# build_vectorstore.py — build local vector index for semantic search
#
# This version:
#   - Reads semantic chunks directly from Supabase (`fincen_semantic_chunks`)
#   - Optionally enriches them with metadata from `fincen_publications`
#   - Writes vecstore/embeddings.npy + vecstore/metadata.jsonl + vecstore/settings.json
#
# It no longer depends on any local CSV files.

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from supabase_helpers import get_supabase_client  # uses your existing helper

# Allow override via environment variable (for Modal Volume)
DATA_DIR = Path(__file__).parent
_vec_dir_env = os.getenv("VECSTORE_DIR")
if _vec_dir_env:
    VEC_DIR = Path(_vec_dir_env)
else:
    VEC_DIR = DATA_DIR / "vecstore"

# Use the same model as the semantic mapper / search
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Candidate text columns in case schema changes
TEXT_COL_CANDIDATES = ["text", "chunk_text", "page_text", "content"]


@dataclass
class Meta:
    idx: int
    article_name: Optional[str]
    page_number: Optional[int]
    matched_fraud_types: List[str]
    text: str
    # full original/augmented row so the app can access any extra fields
    extra: Dict[str, Any]


def _fetch_table_all(
    table_name: str,
    select: str = "*",
    page_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch ALL rows from a Supabase table using .range() pagination.
    """
    client = get_supabase_client()
    rows: List[Dict[str, Any]] = []
    start = 0

    while True:
        resp = (
            client.table(table_name)
            .select(select)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return rows


def load_chunks_from_supabase() -> pd.DataFrame:
    """
    Load semantic chunks directly from Supabase.

    Requires that fincen_ocr_fraud_mapper.py has already populated
    the `fincen_semantic_chunks` table.
    """
    rows = _fetch_table_all("fincen_semantic_chunks", select="*")
    if not rows:
        raise RuntimeError(
            "No rows found in `fincen_semantic_chunks`. "
            "Run the mapper (fincen_ocr_fraud_mapper.py) first."
        )
    df = pd.DataFrame(rows)
    return df


def load_publication_metadata() -> pd.DataFrame:
    """
    Load doc-level metadata from fincen_publications.

    We use this to enrich each chunk with doc_key, date, etc.
    """
    cols = [
        "doc_key",
        "fincen_id",
        "title",
        "date",
        "doc_type",
        "pdf_filename",
        "detail_url",
    ]
    rows = _fetch_table_all("fincen_publications", select=",".join(cols))
    if not rows:
        print("[build_vectorstore] WARNING: fincen_publications has no rows.")
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def _detect_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"Could not detect text column. Looked for: {TEXT_COL_CANDIDATES}. "
        f"Columns available: {list(df.columns)}"
    )


def load_chunks() -> Tuple[pd.DataFrame, str]:
    """
    Main loader: semantic chunks from Supabase + enriched with publication metadata.
    """
    chunks_df = load_chunks_from_supabase()
    text_col = _detect_text_column(chunks_df)

    pub_df = load_publication_metadata()

    # Build lookups by fincen_id (preferred) or by article_name/title
    pub_by_fincen: Dict[str, Dict[str, Any]] = {}
    pub_by_title: Dict[str, Dict[str, Any]] = {}

    if not pub_df.empty:
        for _, row in pub_df.iterrows():
            row_dict = row.to_dict()
            fid = row_dict.get("fincen_id")
            title = (row_dict.get("title") or "").strip()
            if fid:
                pub_by_fincen[str(fid)] = row_dict
            if title:
                pub_by_title[title] = row_dict

    # Enrich each chunk row with doc-level metadata
    enriched_rows: List[Dict[str, Any]] = []
    for _, row in chunks_df.iterrows():
        row_dict = row.to_dict()
        fincen_id = row_dict.get("fincen_id")
        article_name = row_dict.get("article_name")

        meta_row: Dict[str, Any] = {}

        if fincen_id and str(fincen_id) in pub_by_fincen:
            meta_row = pub_by_fincen[str(fincen_id)]
        elif article_name and article_name in pub_by_title:
            meta_row = pub_by_title[article_name]

        # Merge: doc-level metadata first, then chunk fields override
        merged = dict(meta_row)
        merged.update(row_dict)

        enriched_rows.append(merged)

    enriched_df = pd.DataFrame(enriched_rows)
    return enriched_df, text_col


def ensure_vec_dir() -> None:
    VEC_DIR.mkdir(parents=True, exist_ok=True)


def write_metadata(metas: List[Meta]) -> None:
    meta_path = VEC_DIR / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            obj = asdict(m)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[*] Wrote metadata for {len(metas)} chunks → {meta_path}")


def write_embeddings(X: np.ndarray) -> None:
    emb_path = VEC_DIR / "embeddings.npy"
    np.save(emb_path, X.astype("float32"))
    print(f"[*] Wrote embeddings matrix with shape {X.shape} → {emb_path}")


def write_settings(settings: Dict[str, Any]) -> None:
    settings_path = VEC_DIR / "settings.json"
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"[*] Wrote settings → {settings_path}")


def _parse_matched_fraud_types(raw: Any) -> List[str]:
    if raw is None:
        return []
    # Already a list/tuple/set
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw]
    # Try JSON string
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except Exception:
            # Fallback: semicolon or pipe separated
            parts = [p.strip() for p in s.replace("|", ";").split(";")]
            return [p for p in parts if p]
    # Fallback: single value
    return [str(raw)]


def build_metas(df: pd.DataFrame, text_col: str) -> List[Meta]:
    metas: List[Meta] = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        article_name = row_dict.get("article_name")
        page_number = row_dict.get("page_number")

        mft_raw = (
            row_dict.get("fraud_types_semantic")
            or row_dict.get("matched_fraud_types")
            or row_dict.get("primary_fraud_types")
        )
        matched = _parse_matched_fraud_types(mft_raw)

        text = row_dict.get(text_col) or ""
        text = str(text)

        metas.append(
            Meta(
                idx=int(idx),
                article_name=article_name,
                page_number=int(page_number)
                if page_number not in (None, "", "NaN")
                else None,
                matched_fraud_types=matched,
                text=text,
                extra=row_dict,
            )
        )

    return metas


def embed_texts(texts: List[str]) -> np.ndarray:
    print(f"[*] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(texts, batch_size=64, show_progress_bar=True)
    return np.asarray(X)


def main() -> None:
    print("[*] Loading semantic chunks from Supabase…")
    df, text_col = load_chunks()
    print(f"[*] Loaded {len(df)} chunks. Using text column: {text_col!r}")

    ensure_vec_dir()

    metas = build_metas(df, text_col)
    texts = [m.text for m in metas]

    print("[*] Embedding texts…")
    X = embed_texts(texts)

    write_embeddings(X)
    write_metadata(metas)
    write_settings(
        {
            "model": MODEL_NAME,
            "dim": int(X.shape[1]),
            "count": int(X.shape[0]),
            "text_col": text_col,
        }
    )

    print(f"[✓] Vectorstore ready under {VEC_DIR}/")


if __name__ == "__main__":
    main()
