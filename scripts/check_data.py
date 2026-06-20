import os
import json
from typing import List

# Setup Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
REGISTRY_FILE = os.path.join(BASE_DIR, "data", "registry.json")

# Target
TARGET_YEARS: List[int] = [2020, 2021, 2022, 2023]
TARGET_EMITEN: List[str] = [
    "BBCA", "BBRI", "BMRI", "BNGA", "BBNI", "BTPS", "BJTM", "BSDE", "BNII", "MEGA",
    "UNVR", "INDF", "ICBP", "KLBF", "SIDO", "MYOR", "AALI", "SMGR", "GGRM", "HMSP",
    "TLKM", "ISAT", "EXCL", "ADRO", "PTBA", "PGAS", "JSMR", "WIKA", "WSKT", "TOTL"
]

def main() -> None:
    print("\n" + "="*45)
    print("LAPORAN PENGECEKAN DATASET M-RAG")
    print("="*45)

    # 1. Hitung file PDF mentah
    pdf_count = 0
    if os.path.exists(RAW_DIR):
        pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
        pdf_count = len(pdf_files)
        print(f"Total file PDF di data/raw/      : {pdf_count} dokumen")
    else:
        print("Folder data/raw/ tidak ditemukan.")

    # 2. Hitung entri di registry.json
    registry_count = 0
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            try:
                registry = json.load(f)
                registry_count = len(registry)
                print(f"Total entri di registry.json     : {registry_count} dokumen")
            except json.JSONDecodeError:
                print("File registry.json kosong atau rusak.")
    else:
        print("File registry.json tidak ditemukan.")

    # 3. Analisis Kekurangan Data
    total_target = len(TARGET_EMITEN) * len(TARGET_YEARS)
    print(f"\nTotal target dokumen ideal       : {total_target} dokumen (30 emiten x 4 tahun)")
    
    missing_docs = []
    for emiten in TARGET_EMITEN:
        for year in TARGET_YEARS:
            doc_id = f"{emiten}_{year}"
            if doc_id not in registry:
                missing_docs.append(doc_id)

    print(f"Total dokumen yang kosong/gagal  : {len(missing_docs)} dokumen")

    # Print detail jika ada yang kurang
    if missing_docs:
        print("-" * 45)
        print("Daftar dokumen yang tidak ditemukan dari IDX:")
        for i in range(0, len(missing_docs), 5):
            print("  " + ", ".join(missing_docs[i:i+5]))
    print("="*45 + "\n")

if __name__ == "__main__":
    main()