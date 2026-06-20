"""
Script untuk mengekstrak tabel finansial dari PDF Laporan Tahunan.
Menggunakan pencarian kata kunci pintar (PyMuPDF) untuk menemukan lokasi tabel,
lalu mengekstraknya secara akurat menggunakan Camelot.
"""

import os
import json
import logging
import fitz
import camelot
import pandas as pd

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Kata Kunci untuk menemukan halaman tabel keuangan utama
TARGET_KEYWORDS = [
    "laporan posisi keuangan",
    "neraca",
    "laporan laba rugi",
    "laporan arus kas",
    "laporan perubahan ekuitas"
]

def find_financial_statement_pages(pdf_path: str) -> list:
    """Mencari nomor halaman yang kemungkinan besar berisi tabel keuangan."""
    target_pages = set()
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            text = doc[page_num].get_text("text").lower()
            
            # Jika ada kata kunci di halaman ini, catat nomor halamannya (Camelot menggunakan 1-index)
            for keyword in TARGET_KEYWORDS:
                if keyword in text:
                    target_pages.add(page_num + 1)
                    # Tabel sering berlanjut ke halaman berikutnya, tambahkan juga halaman +1
                    if page_num + 1 < doc.page_count:
                        target_pages.add(page_num + 2)
                    break
        doc.close()
    except Exception as e:
        logger.error(f"Gagal memindai halaman untuk {pdf_path}: {e}")
        
    return sorted(list(target_pages))

def extract_tables(pdf_path: str, pages: list) -> list:
    """Mengekstrak tabel dari halaman tertentu menggunakan Camelot."""
    extracted_data = []
    
    if not pages:
        return extracted_data
        
    # Konversi list int ke string koma (contoh: "150,151,155")
    pages_str = ",".join(map(str, pages))
    logger.info(f"    > Menjalankan Camelot pada halaman: {pages_str}")
    
    try:
        # Gunakan flavor='stream' karena tabel laporan keuangan sering tidak bergaris (borderless)
        tables = camelot.read_pdf(pdf_path, pages=pages_str, flavor='stream', split_text=True)
        
        for i, table in enumerate(tables):
            df = table.df
            
            # Bersihkan dataframe: Hapus baris/kolom yang sepenuhnya kosong
            df.replace('', pd.NA, inplace=True)
            df.dropna(how='all', axis=0, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            df.fillna('', inplace=True)
            
            if not df.empty and len(df) > 2: # Pastikan tabel cukup besar (bukan sekadar header)
                # Simpan sebagai list of lists untuk mempertahankan struktur sel
                table_content = df.values.tolist()
                
                extracted_data.append({
                    "table_id": i + 1,
                    "page_number": table.page,
                    "accuracy": table.accuracy,
                    "content": table_content
                })
    except Exception as e:
        logger.error(f"Error Camelot pada {pdf_path}: {e}")
        
    return extracted_data

def main():
    logger.info("Memulai ekstraksi tabel finansial...")
    
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
        
        # 1. Cari halaman potensial (Smart Targeting)
        target_pages = find_financial_statement_pages(pdf_path)
        
        if not target_pages:
            logger.warning(f"    > Kata kunci keuangan tidak ditemukan di {doc_id}.")
            continue
            
        # 2. Ekstrak Tabel
        tables_data = extract_tables(pdf_path, target_pages)
        
        # 3. Simpan Hasil
        if tables_data:
            # Buat sub-folder tables jika ingin lebih rapi, atau satukan di doc_processed_dir
            tables_output_path = os.path.join(doc_processed_dir, "extracted_tables.json")
            with open(tables_output_path, "w", encoding="utf-8") as f:
                json.dump(tables_data, f, indent=4, ensure_ascii=False)
                
            # 4. Update Metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                metadata["jumlah_tabel"] = len(tables_data)
                
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
                    
            logger.info(f"✓ Berhasil mengekstrak {len(tables_data)} tabel krusial untuk {doc_id}")
        else:
            logger.info(f"    > Tidak ada tabel terekstrak dari halaman target untuk {doc_id}")

    logger.info("Ekstraksi tabel selesai!")

if __name__ == "__main__":
    main()