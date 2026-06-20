"""
Script untuk memverifikasi kualitas dan validitas dataset PDF Laporan Tahunan.
Melakukan scanning otomatis pada seluruh file di data/raw/ untuk mendeteksi
anomali (seperti Laporan Keberlanjutan atau dokumen yang terlalu pendek).
"""

import os
import fitz  # PyMuPDF
from colorama import init, Fore, Style

# Inisialisasi pewarnaan terminal
init(autoreset=True)

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def verify_pdf(file_path: str) -> dict:
    """Membaca PDF dan mengembalikan status validitasnya."""
    result = {
        "status": "VALID",
        "reason": "",
        "pages": 0
    }
    
    try:
        doc = fitz.open(file_path)
        result["pages"] = doc.page_count
        
        # 1. Cek jumlah halaman (Laporan Tahunan biasanya > 100 halaman)
        if doc.page_count < 50:
            result["status"] = "INVALID"
            result["reason"] = f"Terlalu tipis ({doc.page_count} hal). Kemungkinan Surat Pengantar."
            doc.close()
            return result
            
        # 2. Ekstrak teks dari 10 halaman pertama
        front_text = ""
        for i in range(min(10, doc.page_count)):
            front_text += doc[i].get_text().lower()
            
        doc.close()
        
        # Jika hasil scan gambar (teks kosong), kita asumsikan valid jika halamannya tebal
        if not front_text.strip():
            result["reason"] = "Teks tidak terdeteksi (PDF Hasil Scan), namun halaman tebal."
            return result
            
        # 3. Deteksi Kata Kunci
        is_sustainability = ("keberlanjutan" in front_text or "sustainability" in front_text)
        is_annual = ("laporan tahunan" in front_text or "annual report" in front_text)
        
        if is_sustainability and not is_annual:
            result["status"] = "INVALID"
            result["reason"] = "Terdeteksi sebagai Laporan Keberlanjutan (Sustainability Report)."
            
    except Exception as e:
        result["status"] = "ERROR"
        result["reason"] = f"File corrupt atau tidak bisa dibaca: {str(e)}"
        
    return result

def main():
    print(f"\n{Style.BRIGHT}=== AUDIT DATASET M-RAG ==={Style.RESET_ALL}")
    
    if not os.path.exists(RAW_DIR):
        print(f"{Fore.RED}Folder {RAW_DIR} tidak ditemukan.{Style.RESET_ALL}")
        return

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"{Fore.YELLOW}Tidak ada file PDF di {RAW_DIR}.{Style.RESET_ALL}")
        return
        
    print(f"Menganalisis {total_files} dokumen...\n")
    
    valid_docs = []
    invalid_docs = []
    error_docs = []
    
    for file_name in pdf_files:
        file_path = os.path.join(RAW_DIR, file_name)
        check = verify_pdf(file_path)
        
        if check["status"] == "VALID":
            valid_docs.append((file_name, check["pages"], check["reason"]))
        elif check["status"] == "INVALID":
            invalid_docs.append((file_name, check["pages"], check["reason"]))
        else:
            error_docs.append((file_name, check["reason"]))
            
    # --- CETAK LAPORAN ---
    print(f"{Fore.GREEN}{Style.BRIGHT}[+] DOKUMEN VALID ({len(valid_docs)}){Style.RESET_ALL}")
    # Hanya tampilkan sampel 5 valid pertama agar terminal tidak penuh
    for doc in valid_docs[:5]:
        note = f" - {doc[2]}" if doc[2] else ""
        print(f"  ✓ {doc[0]} ({doc[1]} halaman){note}")
    if len(valid_docs) > 5:
        print(f"  ... dan {len(valid_docs) - 5} dokumen valid lainnya.")
        
    if invalid_docs:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}[!] DOKUMEN INVALID / SALAH SASARAN ({len(invalid_docs)}){Style.RESET_ALL}")
        for doc in invalid_docs:
            print(f"  ✗ {doc[0]} | {Fore.YELLOW}{doc[2]}{Style.RESET_ALL}")
            
    if error_docs:
        print(f"\n{Fore.RED}{Style.BRIGHT}[x] DOKUMEN CORRUPT / ERROR ({len(error_docs)}){Style.RESET_ALL}")
        for doc in error_docs:
            print(f"  ! {doc[0]} | {doc[1]}")

    # --- RINGKASAN ---
    success_rate = (len(valid_docs) / total_files) * 100
    print(f"\n{Style.BRIGHT}=== KESIMPULAN ==={Style.RESET_ALL}")
    print(f"Total PDF Dievaluasi : {total_files}")
    print(f"Tingkat Presisi Data : {Fore.CYAN}{success_rate:.2f}%{Style.RESET_ALL}")
    print("=" * 25 + "\n")

if __name__ == "__main__":
    main()