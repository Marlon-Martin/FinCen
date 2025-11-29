# modal_app.py
"""
Modal entrypoint for the FinCEN fraud project.

It gives you:
  - run_full_pipeline(): crawler -> text extractor -> summary generator -> mapper -> vecstore
  - build_vectorstore_modal(): build vecstore/ for semantic search
  - serve_streamlit(): run the Streamlit dashboard on Modal

Usage examples (from your repo root: fincen-streamlit-cloud):

  modal run modal_app.py::run_crawler
  modal run modal_app.py::run_full_pipeline
  modal run modal_app.py::build_vectorstore_modal
  modal serve modal_app.py::serve_streamlit
"""

import os
import subprocess
from pathlib import Path

import modal
import time

REPO_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------
# Modal image: install deps + add your project directory
# ---------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # core dependencies from your pyproject
        "bs4",
        "google-genai",
        "pillow",
        "pymupdf",
        "python-dotenv",
        "requests",
        "sentence-transformers",
        "streamlit",
        "supabase",
    )
    # include the whole repo in the container at /root/fincen
    .add_local_dir(str(REPO_ROOT), remote_path="/root/fincen")
)

app = modal.App("fincen-fraud-analytics")


# Persistent storage for vecstore across container restarts
vecstore_volume = modal.Volume.from_name(
    "fincen-vecstore",  # pick any name
    create_if_missing=True,
)


def _chdir_project_root() -> None:
    """Set working directory inside the container."""
    os.chdir("/root/fincen")


# ---------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_crawler():
    """Run the FinCEN publications crawler inside Modal."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_publications_crawler.py"],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_text_extractor():
    """Run your text extractor that fills fincen_fulltext."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_text_extractor.py"],  # adjust if your filename differs
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_summary_generator():
    """Run the LLM summary generator over new docs."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_summary_generator.py"],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
)
def run_mapper():
    """Run the OCR + semantic fraud mapper (fincen_ocr_fraud_mapper.py)."""
    _chdir_project_root()
    subprocess.run(
        ["python", "fincen_ocr_fraud_mapper.py"],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60 * 3,
)
def run_full_pipeline():
    """
    Orchestrate the full pipeline needed for the Streamlit app:

      1) Crawler            → fincen_publications + PDFs in storage
      2) Text extractor     → fincen_fulltext
      3) Summary generator  → fincen_llm_summaries   (Timeline tab)
      4) Mapper             → fincen_semantic_chunks (Semantic tab)
      5) Vecstore builder   → vecstore/ for semantic_search.py
    """
    print("=== [1/5] Running crawler ===")
    run_crawler.call()

    print("=== [2/5] Extracting full text ===")
    run_text_extractor.call()

    print("=== [3/5] Generating summaries ===")
    run_summary_generator.call()

    print("=== [4/5] Running semantic mapper ===")
    run_mapper.call()

    print("=== [5/5] Building vecstore ===")
    build_vectorstore_modal.call()

    print("Full pipeline complete.")


# ---------------------------------------------------------------------
# Vecstore builder for semantic search
# ---------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
    volumes={"/root/fincen-vecstore": vecstore_volume},
    env={"VECSTORE_DIR": "/root/fincen-vecstore"},
)
def build_vectorstore_modal():
    """
    Build the local semantic search vectorstore using Supabase data
    and store it in a Modal Volume so it persists across restarts.
    """
    _chdir_project_root()
    subprocess.run(["python", "build_vectorstore.py"], check=True)
    print("Vectorstore built successfully (volume-backed).")



# ---------------------------------------------------------------------
# Streamlit app on Modal
# ---------------------------------------------------------------------

import os
import subprocess
from pathlib import Path

import modal

# ... your image / app / vecstore_volume definitions above ...

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fincen-secrets")],
    timeout=60 * 60,
    volumes={"/root/fincen-vecstore": vecstore_volume},
    env={"VECSTORE_DIR": "/root/fincen-vecstore"},
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def serve_streamlit():
    """
    Serve the Streamlit dashboard on Modal as a web endpoint.
    This must NOT block – we just spawn Streamlit in the background.
    """
    _chdir_project_root()

    emb_path = Path(os.getenv("VECSTORE_DIR", "vecstore")) / "embeddings.npy"
    if not emb_path.exists():
        print("[serve_streamlit] WARNING: vecstore missing in volume.")
        print("Run `modal run modal_app.py::build_vectorstore_modal` first.")

    cmd = [
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8000",
        "--server.address",
        "0.0.0.0",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]

    # IMPORTANT: non-blocking, do NOT use subprocess.run here
    subprocess.Popen(cmd)




# ---------------------------------------------------------------------
# Local entrypoint (optional helper)
# ---------------------------------------------------------------------


@app.local_entrypoint()
def main(run: str = "pipeline"):
    """
    Convenience for local testing via Modal:

      modal run modal_app.py::main -- --run pipeline
      modal run modal_app.py::main -- --run crawler
      modal run modal_app.py::main -- --run text
      modal run modal_app.py::main -- --run summaries
      modal run modal_app.py::main -- --run mapper
      modal run modal_app.py::main -- --run streamlit
      modal run modal_app.py::main -- --run build_vec
    """
    if run == "crawler":
        run_crawler.call()
    elif run == "text":
        run_text_extractor.call()
    elif run == "summaries":
        run_summary_generator.call()
    elif run == "mapper":
        run_mapper.call()
    elif run == "pipeline":
        run_full_pipeline.call()
    elif run == "streamlit":
        serve_streamlit.call()
    elif run == "build_vec":
        build_vectorstore_modal.call()
    else:
        print(
            "Unknown mode. Use one of: crawler, text, summaries, "
            "mapper, pipeline, streamlit, build_vec"
        )
