"""
Script untuk menserialisasi tabel finansial menjadi representasi teks terstruktur,
lalu melakukan embedding dan menyimpannya ke dalam ChromaDB (gabung dengan teks).
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

# Konfigurasi Model & Batch (Gunakan model yang sama dengan text_embedder)
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 32

def serialize_table(table_data: list, emiten: str, tahun: str) -> str:
    """
    Mengubah struktur list of lists (tabel 2D) menjadi teks deskriptif
    agar model embedding bisa memahami relasi antar sel.
    """
    if not table_data or len(table_data) < 2:
        return ""

    # Asumsi baris pertama adalah header/nama kolom
    headers = [str(h).replace('\n', ' ').strip() for h in table_data[0]]
    kolom_str = ", ".join([h for h in headers if h])
    
    serialized_lines = [f"Tabel Finansial Emiten {emiten} Tahun {tahun}."]
    serialized_lines.append(f"Kolom: {kolom_str}.")
    
    # Iterasi untuk baris data
    for i, row in enumerate(table_data[1:]):
        row_clean = [str(cell).replace('\n', ' ').strip() for cell in row]
        # Gabungkan nama header dengan nilai di baris tersebut
        row_pairs = []
        for j, val in enumerate(row_clean):
            if j < len(headers) and val:
                header_name = headers[j] if headers[j] else f"Kolom_{j+1}"
                row_pairs.append(f"{header_name}: {val}")
        
        if row_pairs:
            serialized_lines.append(f"Baris {i+1}: " + ", ".join(row_pairs) + ".")

    return " ".join(serialized_lines)

def sanitize_metadata(metadata: dict) -> dict:
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)

def process_table_embeddings():
    logger.info("Memulai proses Table Embedding...")
    
    logger.info(f"Memuat model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    # Gunakan collection yang SAMA dengan teks agar hybrid search mudah
    collection = chroma_client.get_or_create_collection(
        name="idx_text",
        metadata={"hnsw:space": "cosine"}
    )

    if not PROCESSED_DIR.exists():
        logger.error(f"Folder {PROCESSED_DIR} tidak ditemukan!")
        return

    doc_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]
    
    for doc_dir in doc_folders:
        tables_file = doc_dir / "extracted_tables.json"
        if not tables_file.exists():
            continue
            
        logger.info(f"Memproses tabel dari: {doc_dir.name}")
        
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        
        with open(tables_file, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
            
            for tbl in tables_data:
                # Gunakan table_id jika ada, atau buat format fallback
                tbl_id = str(tbl.get("table_id", f"{doc_dir.name}_tbl_unknown"))
                if not tbl_id.startswith(doc_dir.name):
                    tbl_id = f"{doc_dir.name}_tbl_{tbl_id}"
                
                emiten = tbl.get("emiten", doc_dir.name.split("_")[0])
                tahun = str(tbl.get("tahun", doc_dir.name.split("_")[1] if "_" in doc_dir.name else ""))
                
                # Serialisasi konten tabel
                content = tbl.get("content", [])
                serialized_text = serialize_table(content, emiten, tahun)
                
                if not serialized_text:
                    continue
                    
                batch_ids.append(tbl_id)
                batch_documents.append(serialized_text)
                
                # Bersihkan metadata (jangan masukkan list data_json secara utuh ke Chroma)
                meta = tbl.copy()
                meta.pop("content", None)
                meta.pop("data_json", None) 
                # Pastikan modalitas tercatat sebagai tabel
                meta["modalitas"] = "tabel"
                
                batch_metadatas.append(sanitize_metadata(meta))
                
                if len(batch_documents) >= BATCH_SIZE:
                    embeddings = model.encode(batch_documents, show_progress_bar=False)
                    normalized_embeddings = normalize_l2(embeddings)
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_documents,
                        embeddings=normalized_embeddings.tolist(),
                        metadatas=batch_metadatas
                    )
                    batch_ids, batch_documents, batch_metadatas = [], [], []
            
            if batch_documents:
                embeddings = model.encode(batch_documents, show_progress_bar=False)
                normalized_embeddings = normalize_l2(embeddings)
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=normalized_embeddings.tolist(),
                    metadatas=batch_metadatas
                )
                
        logger.info(f"  Selesai meng-embed tabel dari {doc_dir.name}")

    logger.info(f"Semua tabel diserialisasi dan di-embed! Total gabungan item di 'idx_text': {collection.count()}")

if __name__ == "__main__":
    process_table_embeddings()