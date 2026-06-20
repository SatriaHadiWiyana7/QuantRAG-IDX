"""
Script Ekstraksi Gambar Vektor & Raster untuk Laporan Keuangan.
Mendeteksi grafik/chart yang digambar menggunakan vektor (garis/path)
dan memfilter gambar raster yang tidak relevan.
"""

import os
import json
import logging
import io
import fitz  # PyMuPDF
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def merge_rects(rects, tolerance=30):
    """
    Menggabungkan kotak pembatas (bounding boxes) garis vektor yang berdekatan.
    Algoritma ini akan menyatukan elemen-elemen penyusun grafik (batang, teks, sumbu) 
    menjadi satu area grafik yang utuh.
    """
    merged = [fitz.Rect(r) for r in rects]
    changed = True
    
    while changed:
        changed = False
        new_merged = []
        while merged:
            current = merged.pop(0)
            # Perlebar kotak untuk mengecek apakah bersentuhan dengan elemen lain
            c_test = current + (-tolerance, -tolerance, tolerance, tolerance)
            
            intersected = []
            for i, other in enumerate(merged):
                o_test = other + (-tolerance, -tolerance, tolerance, tolerance)
                if c_test.intersects(o_test):
                    intersected.append(i)
            
            if intersected:
                changed = True
                # Gabungkan kotak saat ini dengan semua yang bersentuhan
                for i in intersected:
                    current |= merged[i]
                
                # Hapus yang sudah digabung dari daftar
                for i in sorted(intersected, reverse=True):
                    merged.pop(i)
                merged.append(current) 
                break 
            else:
                new_merged.append(current)
                
        if not changed:
            merged = new_merged
            
    return merged

def extract_images(pdf_path: str, doc_id: str, output_image_dir: str) -> list:
    extracted_images_info = []
    
    try:
        doc = fitz.open(pdf_path)
        img_counter = 0
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # ==========================================
            # 1. EKSTRAKSI GRAFIK VEKTOR (Charts/Diagrams)
            # ==========================================
            paths = page.get_drawings()
            if paths:
                rects = []
                for p in paths:
                    # Abaikan titik kecil atau garis background yang memenuhi halaman penuh
                    if p["rect"].width > 5 and p["rect"].height > 5:
                        if p["rect"].width < page.rect.width * 0.95 and p["rect"].height < page.rect.height * 0.95:
                            rects.append(p["rect"])
                
                clustered_bboxes = merge_rects(rects, tolerance=30)
                
                for bbox in clustered_bboxes:
                    # Area Vektor harus cukup besar (mencegah potongan garis acak terekstrak)
                    if bbox.width > 150 and bbox.height > 150:
                        img_counter += 1
                        image_filename = f"{doc_id}_page{page_num+1}_vector_chart_{img_counter}.png"
                        image_filepath = os.path.join(output_image_dir, image_filename)
                        
                        # Render area grafik tersebut menjadi gambar PNG (150 DPI)
                        pix = page.get_pixmap(clip=bbox, dpi=150)
                        pix.save(image_filepath)
                        
                        extracted_images_info.append({
                            "image_id": image_filename,
                            "page_number": page_num + 1,
                            "file_path": image_filepath,
                            "type": "vector_chart",
                            "width": pix.width,
                            "height": pix.height,
                            "format": "PNG",
                            "dpi": 150
                        })

            # ==========================================
            # 2. EKSTRAKSI GAMBAR RASTER (Foto/Logo)
            # ==========================================
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Filter ketat untuk menolak logo dan garis pemisah dekoratif
                    if width <= 150 or height <= 150:
                        continue
                        
                    aspect_ratio = max(width / height, height / width)
                    if aspect_ratio > 5: # Diperketat menjadi 5:1 agar tidak menangkap banner panjang
                        continue
                        
                    img_counter += 1
                    image_filename = f"{doc_id}_page{page_num+1}_raster_{img_counter}.png"
                    image_filepath = os.path.join(output_image_dir, image_filename)
                    
                    if image.mode not in ('RGB', 'RGBA'):
                        image = image.convert('RGB')
                        
                    image.save(image_filepath, format="PNG", dpi=(150, 150))
                    
                    extracted_images_info.append({
                        "image_id": image_filename,
                        "page_number": page_num + 1,
                        "file_path": image_filepath,
                        "type": "raster_photo",
                        "width": width,
                        "height": height,
                        "format": "PNG",
                        "dpi": 150
                    })
                except Exception:
                    pass
                    
        doc.close()
    except Exception as e:
        logger.error(f"Gagal memproses {pdf_path}: {e}")
        
    return extracted_images_info

def main():
    logger.info("Memulai ekstraksi Vektor Grafik & Raster...")
    
    if not os.path.exists(RAW_DIR):
        return

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    
    for file_name in pdf_files:
        doc_id = file_name.replace("_annual_report.pdf", "")
        pdf_path = os.path.join(RAW_DIR, file_name)
        doc_processed_dir = os.path.join(PROCESSED_DIR, doc_id)
        
        image_output_dir = os.path.join(doc_processed_dir, "images")
        os.makedirs(image_output_dir, exist_ok=True)
        
        logger.info(f"Scanning Vektor & Raster: {doc_id}")
        
        images_info = extract_images(pdf_path, doc_id, image_output_dir)
        
        if images_info:
            registry_path = os.path.join(doc_processed_dir, "extracted_images.json")
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(images_info, f, indent=4, ensure_ascii=False)
                
            logger.info(f"✓ {len(images_info)} aset visual diamankan untuk {doc_id}")

if __name__ == "__main__":
    main()