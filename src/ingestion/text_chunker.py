import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Mengambil config dari direktori utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Memecah teks panjang menjadi potongan-potongan kecil (chunks) dengan overlap.
    Pendekatan ini menggunakan sliding window berdasarkan estimasi kata.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        
    return chunks

def process_document_chunks(doc_dir: Path) -> None:
    """
    Membaca extracted_text.json dari folder spesifik, memotongnya, 
    dan menyimpannya sebagai text_chunks.jsonl beserta metadatanya.
    """
    text_file = doc_dir / "extracted_text.json"
    metadata_file = doc_dir / "metadata.json"
    output_file = doc_dir / "text_chunks.jsonl"
    
    if not text_file.exists() or not metadata_file.exists():
        logging.warning(f"File teks atau metadata tidak ditemukan di {doc_dir}")
        return

    # Load sumber
    with open(text_file, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
        
    with open(metadata_file, 'r', encoding='utf-8') as f:
        doc_metadata = json.load(f)

    all_chunks = []
    chunk_index = 0
    
    # Asumsi extracted_data adalah list of dict berisi {halaman, teks}
    for item in extracted_data:
        page_num = item.get("page_number", "unknown")
        page_text = item.get("content", "")
        
        text_chunks = chunk_text(page_text)
        
        for idx, chunk_str in enumerate(text_chunks):
            chunk_data = {
                "chunk_id": f"{doc_metadata.get('doc_id')}_chunk_{chunk_index:04d}",
                "doc_id": doc_metadata.get("doc_id"),
                "emiten": doc_metadata.get("emiten"),
                "tahun": doc_metadata.get("tahun"),
                "halaman": page_num,
                "teks": chunk_str,
                "chunk_index": chunk_index,
                "modalitas": "teks"
            }
            all_chunks.append(chunk_data)
            chunk_index += 1

    # Simpan sebagai JSONL (satu baris satu JSON)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    logging.info(f"Berhasil membuat {len(all_chunks)} chunks untuk {doc_metadata.get('doc_id')}")

if __name__ == "__main__":
    logging.info("Memulai proses text chunking untuk semua dokumen...")
    
    # Pastikan direktori processed ada
    if not PROCESSED_DATA_DIR.exists():
        logging.error(f"Folder data processed tidak ditemukan: {PROCESSED_DATA_DIR}")
    else:
        # Ambil semua folder di dalam direktori processed
        doc_folders = [f for f in PROCESSED_DATA_DIR.iterdir() if f.is_dir()]
        
        if not doc_folders:
            logging.warning("Tidak ada folder dokumen yang ditemukan untuk diproses.")
        
        # Loop untuk memproses setiap folder
        for doc_dir in doc_folders:
            logging.info(f"Memeriksa dokumen: {doc_dir.name}")
            process_document_chunks(doc_dir)
            
        logging.info("Selesai membuat text chunks untuk semua folder dokumen!")