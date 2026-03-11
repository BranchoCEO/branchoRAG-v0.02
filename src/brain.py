import branchorag
import os
import time
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
SCAN_PATH = r"C:\Users\grint\Documents\OneDrive\roop_row\MONEYMOINE\Web Training"
MEMORY_FILE = "brancho_memory.json"
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, ~80MB, fast and good for code

def run_brain():
    print("--- BranchoRAG v0.03: The Embedder ---")

    if not os.path.isdir(SCAN_PATH):
        raise FileNotFoundError(f"Scan path not found or is not a folder: {SCAN_PATH}")

    try:
        rag = branchorag.BranchoRAG()

        # --- STEP 1: SCAN ---
        print(f"Reading files in: {SCAN_PATH}...")
        start = time.perf_counter()
        rag.scan_folder(SCAN_PATH)
        elapsed = time.perf_counter() - start
        count = rag.node_count()
        print(f"  Read {count} file(s) in {elapsed:.2f}s.")

        # --- STEP 2: EMBED ---
        print(f"Loading embedding model '{EMBED_MODEL}'...")
        model = SentenceTransformer(EMBED_MODEL)

        contents = rag.get_contents()  # pull all file texts out of Rust in one call
        print(f"Embedding {len(contents)} file(s)... (this may take a moment)")

        start = time.perf_counter()
        # show_progress_bar gives a live tqdm bar during encoding
        embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
        elapsed = time.perf_counter() - start
        print(f"  Embedded in {elapsed:.2f}s.")

        # convert numpy rows → plain Python lists so PyO3 can accept them
        rag.set_embeddings([emb.tolist() for emb in embeddings])

        # --- STEP 3: SAVE ---
        rag.save_memory(MEMORY_FILE)
        print(f"✅ Success: Knowledge + embeddings saved to {MEMORY_FILE}.")

    except Exception as e:
        print(f"❌ BranchoRAG failed: {e}")
        raise

if __name__ == "__main__":
    run_brain()
