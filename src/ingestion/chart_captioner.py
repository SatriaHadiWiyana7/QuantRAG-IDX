"""
Script untuk menghasilkan deskripsi teks (captioning) dari gambar grafik.
Versi Interaktif: Memindai progres secara otomatis, menampilkan status, 
dan memungkinkan pengguna menentukan ukuran batch secara dinamis.
"""

import os
import json
import logging
import sys
import ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Model Vision yang digunakan
VISION_MODEL = "minicpm-v" 

PROMPT_TEMPLATE = """
Anda adalah Senior Analis Keuangan. Ekstrak data dari gambar ini dan berikan output HANYA dalam format JSON.

ATURAN KLASIFIKASI:
1. Jika gambar HANYA berisi foto orang, pemandangan, gedung, atau ilustrasi tanpa data statistik:
   Kembalikan tepat seperti ini:
   {"chart_type": "noise", "caption_generated": "Gambar ilustrasi non-finansial.", "caption_confidence": 1.0}

2. Jika gambar adalah GRAFIK DATA (Bar/Line/Pie) atau TABEL FINANSIAL:
   Anda WAJIB membaca tulisan di dalamnya. Ganti teks "[DESKRIPSI]" dengan kalimat yang memuat:
   - Judul metrik utama.
   - Angka spesifik untuk setiap tahun.
   - Penjelasan tren secara eksplisit (naik, turun, atau stabil).
   
   Format JSON wajib untuk grafik:
   {
     "chart_type": "bar",
     "caption_generated": "[DESKRIPSI DETAIL YANG MEMUAT JUDUL, ANGKA AKURAT, DAN TREN]",
     "caption_confidence": 0.95
   }
"""

def generate_caption(image_path: str):
    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': PROMPT_TEMPLATE,
                'images': [image_path]
            }],
            options={'temperature': 0.0}
        )
        
        response_text = response['message']['content'].strip()
        
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
            
        result = json.loads(response_text)
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
            
        if isinstance(result, dict):
            return result
        else:
            logger.error(f"Format JSON tidak valid: {type(result)}")
            return None
        
    except json.JSONDecodeError:
        logger.error("Gagal mem-parsing output JSON.")
        return None
    except Exception as e:
        logger.error(f"Error pada saat memproses {image_path}: {e}")
        return None

def scan_progress():
    """Memindai seluruh folder untuk menghitung total dan sisa gambar."""
    total_images = 0
    processed_images = 0
    pending_tasks = []

    if not os.path.exists(PROCESSED_DIR):
        return 0, 0, []

    doc_folders = sorted([f for f in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, f))])
    
    for doc_id in doc_folders:
        doc_dir = os.path.join(PROCESSED_DIR, doc_id)
        images_registry_path = os.path.join(doc_dir, "extracted_images.json")
        captions_dir = os.path.join(doc_dir, "captions")
        
        if not os.path.exists(images_registry_path):
            continue
            
        with open(images_registry_path, "r", encoding="utf-8") as f:
            images_info = json.load(f)
            
        for img in images_info:
            total_images += 1
            caption_filename = f"{img['image_id'].split('.')[0]}.json"
            caption_filepath = os.path.join(captions_dir, caption_filename)
            
            if os.path.exists(caption_filepath):
                processed_images += 1
            else:
                # Simpan metadata untuk dikerjakan nanti
                pending_tasks.append({
                    "doc_id": doc_id,
                    "img_info": img,
                    "caption_filepath": caption_filepath,
                    "captions_dir": captions_dir
                })
                
    return total_images, processed_images, pending_tasks

def main():
    logger.info("Memindai direktori data...")
    total, processed, pending_tasks = scan_progress()
    remaining = len(pending_tasks)
    
    # 1. Menampilkan Laporan Progres
    print("\n" + "="*45)
    print(" 📊 STATUS CAPTIONING M-RAG")
    print("="*45)
    print(f" Total Gambar Ditemukan : {total}")
    print(f" Sudah Diproses (Aman)  : {processed}")
    print(f" Sisa Belum Diproses    : {remaining}")
    print("="*45 + "\n")
    
    if remaining == 0:
        logger.info("🎉 Semua gambar dari seluruh dokumen telah selesai di-caption 100%!")
        return

    # 2. Interaksi Pengguna (Input Terminal)
    try:
        user_input = input(f"Berapa gambar yang ingin diproses sesi ini? (Tekan ENTER untuk 50, ketik 'all' untuk semua sisa): ").strip()
        
        if user_input.lower() == 'all':
            limit = remaining
        elif user_input.isdigit() and int(user_input) > 0:
            limit = int(user_input)
        else:
            limit = 50 # Default aman
            
    except KeyboardInterrupt:
        print("\nDibatalkan oleh pengguna.")
        sys.exit(0)
        
    # Memotong daftar tugas sesuai limit yang diminta
    tasks_to_process = pending_tasks[:limit]
    
    logger.info(f"🚀 Memulai pemrosesan {len(tasks_to_process)} gambar menggunakan {VISION_MODEL}...\n")
    
    for i, task in enumerate(tasks_to_process):
        doc_id = task["doc_id"]
        img = task["img_info"]
        captions_dir = task["captions_dir"]
        caption_filepath = task["caption_filepath"]
        caption_filename = os.path.basename(caption_filepath)
        
        os.makedirs(captions_dir, exist_ok=True)
        
        logger.info(f"[{i+1}/{len(tasks_to_process)}] Menganalisis: {img['image_id']} ({doc_id})")
        caption_result = generate_caption(img['file_path'])
        
        if caption_result:
            final_data = {
                "caption_id": caption_filename.replace(".json", ""),
                "doc_id": doc_id,
                "halaman": img['page_number'],
                "image_path": img['file_path'],
                "caption_generated": caption_result.get("caption_generated", ""),
                "caption_confidence": float(caption_result.get("caption_confidence", 0.0)),
                "chart_type": caption_result.get("chart_type", "unknown"),
                "modalitas": "gambar"
            }
            
            with open(caption_filepath, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
                
    logger.info(f"Sesi selesai! {len(tasks_to_process)} gambar berhasil di-caption.")
    logger.info("Jalankan ulang skrip ini kapan saja untuk memproses sisa gambar.")

if __name__ == "__main__":
    main()