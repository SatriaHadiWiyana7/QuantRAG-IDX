"""
Script untuk melakukan embedding pada text chunks menggunakan SentenceTransformers
dan menyimpannya ke dalam ChromaDB (Vector Store).
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Konfigurasi Model & Batch
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64

def sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB hanya menerima string, int, float, atau bool di dalam metadata.
    Fungsi ini akan membersihkan tipe data yang tidak didukung (seperti list/dict).
    """
    clean_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        elif v is None:
            clean_meta[k] = ""
        else:
            clean_meta[k] = str(v)
    return clean_meta

def normalize_l2(embeddings):
    """Melakukan L2-normalization pada embeddings sesuai instruksi panduan."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12) # Menghindari pembagian dengan nol

def process_text_embeddings():
    logger.info("Memulai proses Text Embedding...")
    
    # Inisialisasi Model Embedding
    logger.info(f"Memuat model: {MODEL_NAME} (Ini mungkin memakan waktu pertama kali)")
    model = SentenceTransformer(MODEL_NAME)
    
    # Inisialisasi ChromaDB
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    
    # Buat atau ambil collection khusus teks
    collection = chroma_client.get_or_create_collection(
        name="idx_text",
        metadata={"hnsw:space": "cosine"} # Menggunakan cosine similarity
    )
    logger.info("ChromaDB siap. Collection 'idx_text' diakses.")

    # Cari semua file text_chunks.jsonl
    if not PROCESSED_DIR.exists():
        logger.error(f"Folder {PROCESSED_DIR} tidak ditemukan!")
        return

    doc_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]
    
    for doc_dir in doc_folders:
        chunks_file = doc_dir / "text_chunks.jsonl"
        if not chunks_file.exists():
            continue
            
        logger.info(f"Memproses embedding untuk: {doc_dir.name}")
        
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                chunk = json.loads(line)
                
                batch_ids.append(chunk["chunk_id"])
                batch_documents.append(chunk["teks"])
                
                # Buang teks dari metadata untuk menghemat space (sudah ada di document)
                meta = chunk.copy()
                meta.pop("teks", None)
                batch_metadatas.append(sanitize_metadata(meta))
                
                # Jika batch sudah penuh, proses dan simpan
                if len(batch_documents) >= BATCH_SIZE:
                    # Generate embeddings
                    embeddings = model.encode(batch_documents, show_progress_bar=False)
                    # L2 Normalize
                    normalized_embeddings = normalize_l2(embeddings)
                    
                    # Simpan ke ChromaDB
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_documents,
                        embeddings=normalized_embeddings.tolist(),
                        metadatas=batch_metadatas
                    )
                    
                    # Kosongkan batch
                    batch_ids, batch_documents, batch_metadatas = [], [], []
            
            # Proses sisa dokumen di batch terakhir
            if batch_documents:
                embeddings = model.encode(batch_documents, show_progress_bar=False)
                normalized_embeddings = normalize_l2(embeddings)
                
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=normalized_embeddings.tolist(),
                    metadatas=batch_metadatas
                )
                
        logger.info(f"  Selesai meng-embed teks dari {doc_dir.name}")

    logger.info(f"Semua teks berhasil di-embed dan disimpan ke Vector Store! Total item di collection: {collection.count()}")

if __name__ == "__main__":
    process_text_embeddings()