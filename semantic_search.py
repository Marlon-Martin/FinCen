# semantic_search.py — simple cosine similarity semantic search over vecstore

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent

_vec_dir_env = os.getenv("VECSTORE_DIR")
if _vec_dir_env:
    VEC_DIR = Path(_vec_dir_env)
else:
    VEC_DIR = DATA_DIR / "vecstore"

SETTINGS_PATH = VEC_DIR / "settings.json"
EMB_PATH = VEC_DIR / "embeddings.npy"
META_PATH = VEC_DIR / "metadata.jsonl"


def load_settings() -> Dict:
    if not SETTINGS_PATH.exists():
        raise FileNotFoundError(
            f"{SETTINGS_PATH} not found. Run build_vectorstore.py first."
        )
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def load_embeddings() -> np.ndarray:
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run build_vectorstore.py first."
        )
    X = np.load(EMB_PATH)
    # shape: (N, D)
    return X.astype("float32")


def load_metadata() -> List[Dict]:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} not found. Run build_vectorstore.py first."
        )
    out: List[Dict] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    print(f"[*] Loading semantic search model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _get_index() -> tuple[np.ndarray, List[Dict], Dict]:
    settings = load_settings()
    X = load_embeddings()
    metas = load_metadata()
    if X.shape[0] != len(metas):
        raise RuntimeError(
            f"Embedding count {X.shape[0]} != metadata count {len(metas)}. "
            "Rebuild the vectorstore."
        )
    return X, metas, settings


def _cosine_sim_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    # q: (D,), X: (N, D) -> sims: (N,)
    q = q.astype("float32")
    X = X.astype("float32")
    q_norm = np.linalg.norm(q) + 1e-8
    X_norm = np.linalg.norm(X, axis=1) + 1e-8
    sims = (X @ q) / (X_norm * q_norm)
    return sims


def search(
    query: str,
    *,
    k: int = 5,
    label: Optional[str] = None,
    article: Optional[str] = None,
) -> List[Dict]:
    """
    Basic semantic search over chunks.

    - query: user question
    - k: top-k results to return
    - label: optional fraud label filter; matched against any of:
        * meta["matched_fraud_types"]
        * meta["extra"].get("fraud_type")
    - article: optional substring match against meta["article_name"]
    """
    if not query.strip():
        return []

    X, metas, settings = _get_index()
    model = _get_model(settings["model"])

    q_vec = model.encode([query], convert_to_numpy=True)[0].astype("float32")
    sims = _cosine_sim_matrix(q_vec, X)

    # Optional filtering by label / article
    cand_indices = np.arange(len(metas))

    if label:
        label_lower = label.lower()
        mask = []
        for i in cand_indices:
            m = metas[i]
            labels = []

            mft = m.get("matched_fraud_types") or []
            if isinstance(mft, str):
                # try JSON or semicolon list
                try:
                    parsed = json.loads(mft)
                    if isinstance(parsed, list):
                        labels.extend([str(x) for x in parsed])
                    else:
                        labels.append(str(parsed))
                except Exception:
                    labels.extend([s.strip() for s in mft.split(";") if s.strip()])
            elif isinstance(mft, (list, tuple, set)):
                labels.extend([str(x) for x in mft])

            fraud_type = (m.get("extra") or {}).get("fraud_type")
            if fraud_type:
                labels.append(str(fraud_type))

            labels_lower = [l.lower() for l in labels]
            mask.append(any(label_lower in l for l in labels_lower))

        cand_indices = cand_indices[np.array(mask)]

    if article:
        article_lower = article.lower()
        mask = []
        for i in cand_indices:
            name = metas[i].get("article_name") or ""
            mask.append(article_lower in str(name).lower())
        cand_indices = cand_indices[np.array(mask)]

    # If no candidates after filtering, short-circuit
    if len(cand_indices) == 0:
        return []

    sims_cand = sims[cand_indices]
    top_idx = np.argsort(-sims_cand)[:k]
    results: List[Dict] = []

    for rank_pos in top_idx:
        global_idx = int(cand_indices[rank_pos])
        m = metas[global_idx]
        r = dict(m)  # shallow copy
        r["similarity"] = float(sims[global_idx])
        results.append(r)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Ad-hoc CLI semantic search")
    ap.add_argument("query", type=str, help="Search query")
    ap.add_argument("--k", type=int, default=5, help="Top-k results")
    ap.add_argument("--label", type=str, default=None, help="Optional fraud label filter")
    ap.add_argument("--article", type=str, default=None, help="Optional article filter")
    args = ap.parse_args()

    results = search(args.query, k=args.k, label=args.label, article=args.article)
    for r in results:
        name = r.get("article_name")
        page = r.get("page_number")
        labels = r.get("matched_fraud_types") or []
        if isinstance(labels, str):
            labels = [labels]
        print(
            f"{r['similarity']:.4f} | {name} p.{page} | labels={','.join(map(str, labels))}\n"
            f"{r['text']}\n---"
        )


if __name__ == "__main__":
    main()
