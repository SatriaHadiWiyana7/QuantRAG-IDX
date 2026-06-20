"""
Script untuk membangun dan memperkaya metadata pada semua chunk (teks, tabel, gambar)
sesuai dengan PANDUAN_MRAG_LAPKEU_IDX.md sebelum dimasukkan ke Vector Store.
"""

import os
import json
import logging
from pathlib import Path

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Mapping Sektor sesuai stratifikasi di panduan
SEKTOR_MAP = {
    # Perbankan
    "BBCA": "perbankan", "BBRI": "perbankan", "BMRI": "perbankan", "BNGA": "perbankan",
    "BBNI": "perbankan", "BTPS": "perbankan", "BJTM": "perbankan", "BSDE": "perbankan",
    "BNII": "perbankan", "MEGA": "perbankan",
    # Manufaktur/FMCG
    "UNVR": "manufaktur", "INDF": "manufaktur", "ICBP": "manufaktur", "KLBF": "manufaktur",
    "SIDO": "manufaktur", "MYOR": "manufaktur", "AALI": "manufaktur", "SMGR": "manufaktur",
    "GGRM": "manufaktur", "HMSP": "manufaktur",
    # Infrastruktur/Energi
    "TLKM": "infrastruktur", "ISAT": "infrastruktur", "EXCL": "infrastruktur", "ADRO": "infrastruktur",
    "PTBA": "infrastruktur", "PGAS": "infrastruktur", "JSMR": "infrastruktur", "WIKA": "infrastruktur",
    "WSKT": "infrastruktur", "TOTL": "infrastruktur"
}

def get_sektor(emiten: str) -> str:
    """Mendapatkan sektor berdasarkan kode emiten."""
    return SEKTOR_MAP.get(emiten.upper(), "lainnya")

def estimate_tokens(text: str) -> int:
    """Estimasi jumlah token (1 kata bahasa Indonesia ~ 1.3 token)."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def get_posisi_dokumen(halaman: int, total_halaman: int) -> str:
    """Menentukan posisi chunk di dalam dokumen (awal/tengah/akhir)."""
    if total_halaman <= 0 or halaman <= 0:
        return "unknown"
    ratio = halaman / total_halaman
    if ratio <= 0.33:
        return "awal"
    elif ratio <= 0.66:
        return "tengah"
    else:
        return "akhir"

def extract_section_heuristic(text: str) -> str:
    """Heuristik sederhana untuk mendeteksi section dari awal teks."""
    # Ambil 5 kata pertama untuk dicek apakah terlihat seperti header
    words = text.split()[:5]
    header_candidate = " ".join(words).upper()
    
    if "DIREKSI" in header_candidate:
        return "Laporan Direksi"
    elif "KOMISARIS" in header_candidate:
        return "Laporan Dewan Komisaris"
    elif "KEUANGAN" in header_candidate or "NERACA" in header_candidate:
        return "Laporan Keuangan"
    elif "TATA KELOLA" in header_candidate:
        return "Tata Kelola Perusahaan"
    
    return "unknown"

def process_metadata():
    logger.info("Memulai proses pembangunan dan pengayaan Metadata M-RAG...")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Folder {PROCESSED_DIR} tidak ditemukan!")
        return

    doc_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]
    
    for doc_dir in doc_folders:
        logger.info(f"Memproses: {doc_dir.name}")
        metadata_file = doc_dir / "metadata.json"
        
        # 1. Pastikan dan Perkaya metadata.json Utama
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                doc_metadata = json.load(f)
        else:
            # Jika belum ada, buat metadata dasar dari nama folder
            parts = doc_dir.name.split("_")
            emiten = parts[0] if len(parts) > 0 else "UNKNOWN"
            tahun = parts[1] if len(parts) > 1 else "0000"
            doc_metadata = {
                "doc_id": doc_dir.name,
                "emiten": emiten,
                "tahun": int(tahun) if tahun.isdigit() else tahun,
                "total_halaman": 0
            }

        emiten = doc_metadata.get("emiten", "")
        sektor = get_sektor(emiten)
        total_halaman = doc_metadata.get("total_halaman", 0)
        
        doc_metadata["sektor"] = sektor
        
        # Simpan kembali metadata.json
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(doc_metadata, f, indent=4)

        # 2. Perkaya text_chunks.jsonl
        text_chunks_file = doc_dir / "text_chunks.jsonl"
        if text_chunks_file.exists():
            enriched_chunks = []
            with open(text_chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    chunk = json.loads(line)
                    
                    # Injeksi Metadata Wajib
                    chunk["sektor"] = sektor
                    chunk["token_count"] = estimate_tokens(chunk.get("teks", ""))
                    chunk["posisi_dalam_dokumen"] = get_posisi_dokumen(chunk.get("halaman", 0), total_halaman)
                    
                    if chunk.get("section", "unknown") == "unknown":
                        chunk["section"] = extract_section_heuristic(chunk.get("teks", ""))
                        
                    enriched_chunks.append(chunk)
                    
            # Tulis ulang file JSONL dengan metadata baru
            with open(text_chunks_file, "w", encoding="utf-8") as f:
                for chunk in enriched_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
            logger.info(f"  ✓ {len(enriched_chunks)} Teks chunks diperbarui dengan metadata presisi.")
            
        # 3. Perkaya extracted_tables.json (jika ada)
        tables_file = doc_dir / "extracted_tables.json"
        if tables_file.exists():
            with open(tables_file, "r", encoding="utf-8") as f:
                tables_data = json.load(f)
                
            for tbl in tables_data:
                tbl["doc_id"] = doc_dir.name
                tbl["emiten"] = emiten
                tbl["tahun"] = doc_metadata.get("tahun")
                tbl["sektor"] = sektor
                tbl["posisi_dalam_dokumen"] = get_posisi_dokumen(tbl.get("page_number", 0), total_halaman)
                tbl["modalitas"] = "tabel"
                
            with open(tables_file, "w", encoding="utf-8") as f:
                json.dump(tables_data, f, indent=4, ensure_ascii=False)
            logger.info(f"  ✓ Metadata ditambahkan pada {len(tables_data)} Tabel.")

    logger.info("Pembangunan Metadata M-RAG selesai! Data siap masuk ke Vector Indexing.")

if __name__ == "__main__":
    process_metadata()