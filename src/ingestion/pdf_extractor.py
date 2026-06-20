"""
Script untuk mengekstrak teks naratif dari dokumen PDF Laporan Tahunan.
Menyimpan hasil ekstraksi dalam format JSON per dokumen dan memperbarui
jumlah halaman di metadata.json.
"""

import os
import json
import logging
import fitz  # PyMuPDF
from typing import Dict, Any

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def clean_text(text: str) -> str:
    """Membersihkan teks dari karakter yang tidak perlu."""
    if not text:
        return ""
    # Hapus spasi ganda dan karakter newline yang berlebihan
    cleaned = " ".join(text.split())
    return cleaned

def extract_text_from_pdf(pdf_path: str) -> list:
    """Mengekstrak teks per halaman dari PDF."""
    extracted_pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            
            extracted_pages.append({
                "page_number": page_num + 1,
                "content": cleaned_text,
                "char_count": len(cleaned_text)
            })
        doc.close()
    except Exception as e:
        logger.error(f"Gagal mengekstrak teks dari {pdf_path}: {e}")
        
    return extracted_pages

def main():
    logger.info("Memulai proses ekstraksi teks naratif dari PDF...")
    
    if not os.path.exists(RAW_DIR):
        logger.error(f"Folder {RAW_DIR} tidak ditemukan!")
        return

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    
    for file_name in pdf_files:
        doc_id = file_name.replace("_annual_report.pdf", "")
        pdf_path = os.path.join(RAW_DIR, file_name)
        doc_processed_dir = os.path.join(PROCESSED_DIR, doc_id)
        metadata_path = os.path.join(doc_processed_dir, "metadata.json")
        
        logger.info(f"Memproses: {doc_id}")
        
        # 1. Ekstrak Teks
        pages_data = extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            logger.warning(f"Teks kosong untuk {doc_id}")
            continue
            
        # 2. Simpan Teks ke JSON
        text_output_path = os.path.join(doc_processed_dir, "extracted_text.json")
        with open(text_output_path, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, indent=4, ensure_ascii=False)
            
        # 3. Update Metadata (total_halaman)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            metadata["total_halaman"] = len(pages_data)
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
        logger.info(f"✓ Berhasil mengekstrak {len(pages_data)} halaman untuk {doc_id}")

    logger.info("Ekstraksi teks naratif selesai!")

if __name__ == "__main__":
    main()