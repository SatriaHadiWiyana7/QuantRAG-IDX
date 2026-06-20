"""
Script Lapis Baja untuk mengunduh Laporan Tahunan (Annual Report) IDX.
Dilengkapi Validasi Ganda: 
1. Pengecekan Kode Emiten dari JSON (mencegah salah perusahaan).
2. Pengecekan Konten PDF via PyMuPDF (mencegah Laporan Keberlanjutan & Surat).
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any

from curl_cffi import requests
import fitz  # PyMuPDF

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Konfigurasi Target Emiten
TARGET_YEARS: List[int] = [2020, 2021, 2022, 2023]
TARGET_EMITEN: Dict[str, str] = {
    "BBCA": "perbankan", "BBRI": "perbankan", "BMRI": "perbankan", "BNGA": "perbankan", 
    "BBNI": "perbankan", "BTPS": "perbankan", "BJTM": "perbankan", "BSDE": "perbankan", 
    "BNII": "perbankan", "MEGA": "perbankan",
    "UNVR": "manufaktur", "INDF": "manufaktur", "ICBP": "manufaktur", "KLBF": "manufaktur", 
    "SIDO": "manufaktur", "MYOR": "manufaktur", "AALI": "manufaktur", "SMGR": "manufaktur", 
    "GGRM": "manufaktur", "HMSP": "manufaktur",
    "TLKM": "infrastruktur", "ISAT": "infrastruktur", "EXCL": "infrastruktur", "ADRO": "infrastruktur", 
    "PTBA": "infrastruktur", "PGAS": "infrastruktur", "JSMR": "infrastruktur", "WIKA": "infrastruktur", 
    "WSKT": "infrastruktur", "TOTL": "infrastruktur"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REGISTRY_FILE = os.path.join(BASE_DIR, "data", "registry.json")

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan/",
}

def setup_directories() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "w") as f: json.dump({}, f)

def get_md5_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""

def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {}

def save_registry(registry_data: Dict[str, Any]) -> None:
    with open(REGISTRY_FILE, "w") as f: json.dump(registry_data, f, indent=4)

def is_valid_annual_report(pdf_path: str) -> bool:
    """Inspeksi dokumen untuk membedakan Annual Report dari Laporan Keberlanjutan/Surat."""
    try:
        doc = fitz.open(pdf_path)
        
        # Aturan 1: Laporan tahunan pasti tebal (Bypass surat pengantar yang biasanya < 50 halaman)
        if doc.page_count < 50:
            doc.close()
            return False
            
        # Ekstrak teks dari 5 halaman pertama untuk inspeksi
        front_text = ""
        for i in range(min(5, doc.page_count)):
            front_text += doc[i].get_text().lower()
            
        doc.close()
        
        if not front_text.strip():
            return True # Loloskan jika berupa gambar hasil scan tapi tebal halamannya
            
        # Aturan 2: Deteksi Laporan Keberlanjutan murni
        is_sustainability = ("keberlanjutan" in front_text or "sustainability" in front_text)
        is_annual = ("laporan tahunan" in front_text or "annual report" in front_text)
        
        if is_sustainability and not is_annual:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Gagal memvalidasi PDF: {e}")
        return False

def main() -> None:
    setup_directories()
    registry = load_registry()
    session = requests.Session(impersonate="chrome120")
    session.headers.update(HEADERS)
    
    logger.info("Memulai scraper (Validasi Ganda: JSON & PDF Content)...")
    
    for emiten, sektor in TARGET_EMITEN.items():
        for year in TARGET_YEARS:
            doc_id = f"{emiten}_{year}"
            if doc_id in registry:
                continue
                
            logger.info(f"Mencari data: {doc_id}...")
            
            # Gunakan parameter asli (emitent) yang dikombinasikan
            api_urls = [
                # --- FORMAT BARU ---
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=20&Year={year}&ReportType=Tahunan&KodeEmiten={emiten}",
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=20&Year={year}&Periode=Audit&KodeEmiten={emiten}",
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=50&Year={year}&KodeEmiten={emiten}",
                
                # --- FORMAT LAMA ---
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=20&year={year}&reportType=Tahunan&emitent={emiten}",
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=20&year={year}&period=Audit&emitent={emiten}",
                f"https://www.idx.co.id/primary/ListedCompany/GetFinancialReport?indexFrom=1&pageSize=50&year={year}&emitent={emiten}"
            ]
            
            final_pdf_url = None
            nama_perusahaan = emiten
            
            for api_url in api_urls:
                try:
                    response = session.get(api_url, timeout=15)
                    if response.status_code == 403:
                        time.sleep(5)
                        continue
                        
                    data = response.json()
                    
                    if "Results" in data and len(data["Results"]) > 0:
                        for report in data["Results"]:
                            
                            # ==============================================================
                            # VALIDASI LAPIS 1: Pastikan ini dokumen emiten yang kita cari!
                            # Mencegah anomali server IDX yang me-return PT Mahaka (ABBA) dll.
                            # ==============================================================
                            api_emiten_code = report.get("Emiten_Code", report.get("KodeEmiten", "")).upper()
                            if api_emiten_code and api_emiten_code != emiten.upper():
                                continue # Lewati data nyasar secara diam-diam
                            
                            for att in report.get("Attachments", []):
                                file_path = att.get("File_Path", "")
                                
                                if file_path.lower().endswith(".pdf"):
                                    full_url = file_path if file_path.startswith("http") else f"https://www.idx.co.id{file_path}"
                                    temp_path = os.path.join(RAW_DIR, f"temp_{doc_id}.pdf")
                                    
                                    logger.info(f"  > Menginspeksi: {full_url.split('/')[-1]}")
                                    
                                    # Unduh sementara
                                    pdf_res = session.get(full_url, stream=True, timeout=30)
                                    with open(temp_path, "wb") as f:
                                        for chunk in pdf_res.iter_content(chunk_size=8192):
                                            if chunk: f.write(chunk)
                                            
                                    # ==============================================================
                                    # VALIDASI LAPIS 2: Baca isi PDF dengan AI / PyMuPDF
                                    # ==============================================================
                                    if is_valid_annual_report(temp_path):
                                        final_pdf_url = full_url
                                        nama_perusahaan = report.get("Emiten_Name", emiten)
                                        
                                        # Simpan permanen
                                        save_path = os.path.join(RAW_DIR, f"{doc_id}_annual_report.pdf")
                                        if os.path.exists(save_path): os.remove(save_path)
                                        os.rename(temp_path, save_path)
                                        break
                                    else:
                                        logger.info("  > [DITOLAK] Bukan Laporan Tahunan (Surat/Keberlanjutan/Singkat).")
                                        os.remove(temp_path)
                                        
                            if final_pdf_url: break
                    if final_pdf_url: break
                except Exception as e:
                    logger.debug(f"Error pada URL {api_url}: {e}")
            
            if final_pdf_url:
                save_path = os.path.join(RAW_DIR, f"{doc_id}_annual_report.pdf")
                doc_metadata = {
                    "doc_id": doc_id, "emiten": emiten, "nama": nama_perusahaan,
                    "tahun": year, "sektor": sektor,
                    "total_halaman": 0, "jumlah_tabel": 0, "jumlah_gambar": 0,
                    "sumber_url": final_pdf_url, "tanggal_unduh": datetime.now().strftime("%Y-%m-%d"),
                    "md5_hash": get_md5_hash(save_path)
                }
                
                os.makedirs(os.path.join(PROCESSED_DIR, doc_id), exist_ok=True)
                with open(os.path.join(PROCESSED_DIR, doc_id, "metadata.json"), "w") as f:
                    json.dump(doc_metadata, f, indent=4)
                
                registry[doc_id] = doc_metadata
                save_registry(registry)
                logger.info(f"SUCCESS - {doc_id} Laporan Tahunan Asli berhasil diamankan.")
            else:
                logger.warning(f"NOT FOUND - Data asli untuk {doc_id} tidak tersedia di server IDX.")
            
            time.sleep(2)

if __name__ == "__main__":
    main()