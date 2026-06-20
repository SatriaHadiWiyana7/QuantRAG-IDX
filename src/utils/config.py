"""
Konfigurasi terpusat untuk proyek M-RAG Laporan Keuangan IDX.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ingestion & Chunking Settings
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

# Indexing & Vector Store Settings
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
VISION_MODEL = "openai/clip-vit-large-patch14"
CHROMA_DB_DIR = BASE_DIR / "vector_store"
COLLECTION_TEXT = "idx_text"
COLLECTION_IMAGE = "idx_image"
COLLECTION_BM25 = "idx_bm25"

# Retrieval Settings
TOP_K_RETRIEVAL = 10