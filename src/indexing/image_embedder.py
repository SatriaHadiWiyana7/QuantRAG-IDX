"""
Script untuk melakukan embedding pada gambar (grafik/chart) menggunakan model CLIP,
serta menyimpan caption teksnya ke dalam koleksi teks untuk cross-modal linking.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Konfigurasi Model
IMAGE_MODEL_NAME = "clip-ViT-L-14" 
TEXT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 32

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

def process_image_embeddings():
    logger.info("Memulai proses Image & Caption Embedding...")
    
    # Load Models
    logger.info(f"Memuat Vision Model: {IMAGE_MODEL_NAME}")
    image_model = SentenceTransformer(IMAGE_MODEL_NAME)
    
    logger.info(f"Memuat Text Model: {TEXT_MODEL_NAME}")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    
    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    
    # Collection untuk Vektor Gambar Asli
    image_collection = chroma_client.get_or_create_collection(
        name="idx_image",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Collection untuk Teks (gabung dengan narasi dan tabel)
    text_collection = chroma_client.get_or_create_collection(
        name="idx_text",
        metadata={"hnsw:space": "cosine"}
    )

    if not PROCESSED_DIR.exists():
        logger.error(f"Folder {PROCESSED_DIR} tidak ditemukan!")
        return

    doc_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]
    
    for doc_dir in doc_folders:
        captions_dir = doc_dir / "captions"
        if not captions_dir.exists():
            continue
            
        logger.info(f"Memproses gambar dan caption dari: {doc_dir.name}")
        
        caption_files = [f for f in captions_dir.iterdir() if f.suffix == '.json']
        
        for caption_file in caption_files:
            try:
                with open(caption_file, "r", encoding="utf-8") as f:
                    cap_data = json.load(f)
                
                caption_id = cap_data.get("caption_id")
                image_rel_path = cap_data.get("image_path")
                caption_text = cap_data.get("caption_generated", "")
                
                if not caption_id or not image_rel_path:
                    continue
                    
                # Path gambar relatif terhadap base_dir atau doc_dir
                # Menyesuaikan dengan struktur output di image_extractor.py
                image_full_path = doc_dir / image_rel_path if not os.path.isabs(image_rel_path) else Path(image_rel_path)
                
                if not image_full_path.exists():
                    logger.warning(f"File gambar tidak ditemukan: {image_full_path}")
                    continue
                
                # Buka gambar
                img = Image.open(image_full_path)
                
                # --- PROSES EMBEDDING GAMBAR ---
                img_emb = image_model.encode([img], show_progress_bar=False)
                img_emb_norm = normalize_l2(img_emb)
                
                # Siapkan metadata
                meta = cap_data.copy()
                meta.pop("caption_generated", None) # Buang teks panjang dari metadata
                meta["modalitas"] = "gambar"
                clean_meta = sanitize_metadata(meta)
                
                image_collection.upsert(
                    ids=[f"img_{caption_id}"],
                    embeddings=img_emb_norm.tolist(),
                    metadatas=[clean_meta]
                )
                
                # --- PROSES EMBEDDING CAPTION TEKS ---
                if caption_text and cap_data.get("chart_type") != "noise":
                    # Tambahkan prefix agar LLM tahu ini deskripsi dari gambar
                    enhanced_caption = f"Grafik/Chart ({cap_data.get('chart_type')}): {caption_text}"
                    
                    text_emb = text_model.encode([enhanced_caption], show_progress_bar=False)
                    text_emb_norm = normalize_l2(text_emb)
                    
                    # Metadata yang sama agar bisa di-link
                    text_collection.upsert(
                        ids=[f"txtcap_{caption_id}"],
                        documents=[enhanced_caption],
                        embeddings=text_emb_norm.tolist(),
                        metadatas=[clean_meta]
                    )
                    
            except Exception as e:
                logger.error(f"Gagal memproses {caption_file.name}: {e}")
                
        logger.info(f"Selesai meng-embed visual dari {doc_dir.name}")

    logger.info(f"Fase Multi-Index Selesai! Total item di 'idx_image': {image_collection.count()}")

if __name__ == "__main__":
    process_image_embeddings()