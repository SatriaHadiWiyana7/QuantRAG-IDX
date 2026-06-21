"""
Script untuk membangun Prompt LLM secara terstruktur.
Mengambil dokumen hasil reranking, memisahkannya berdasarkan modalitas,
dan memasukkannya ke dalam template instruksi sistem yang ketat anti-halusinasi.
"""

import logging
from typing import List, Dict

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self):
        logger.info("Inisialisasi Prompt Builder...")

    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Membangun prompt lengkap berdasarkan format
        """
        # Pemisahan konteks berdasarkan modalitas
        teks_chunks = []
        tabel_chunks = []
        gambar_captions = []

        for idx, doc in enumerate(retrieved_docs):
            meta = doc.get("metadata", {})
            modalitas = meta.get("modalitas", "unknown").lower()
            emiten = meta.get("emiten", "Unknown")
            hal = meta.get("halaman", "?")
            konten = doc.get("content", "").strip()

            # Format sitasi internal untuk LLM
            formatted_content = f"[Sumber: {emiten}, Hal {hal}]\n{konten}\n"

            if modalitas == "teks":
                teks_chunks.append(formatted_content)
            elif modalitas == "tabel":
                tabel_chunks.append(formatted_content)
            elif modalitas == "gambar":
                # Gambar menggunakan representasi teks/caption-nya
                gambar_captions.append(formatted_content)
            else:
                teks_chunks.append(formatted_content)

        # Menggabungkan list menjadi string
        str_teks = "\n".join(teks_chunks) if teks_chunks else "Tidak ada konteks teks naratif."
        str_tabel = "\n".join(tabel_chunks) if tabel_chunks else "Tidak ada konteks tabel finansial."
        str_gambar = "\n".join(gambar_captions) if gambar_captions else "Tidak ada konteks grafik/chart."

        # Template prompt ketat untuk menghindari halusinasi
        prompt = f"""[SISTEM]
Anda adalah asisten analisis laporan keuangan untuk perusahaan publik Indonesia.
Jawab pertanyaan HANYA berdasarkan konteks yang diberikan.
Jika informasi tidak ada dalam konteks, katakan "Informasi tidak tersedia dalam dokumen."
JANGAN membuat angka atau fakta yang tidak ada dalam konteks.

[KONTEKS TEKS]
{str_teks}

[KONTEKS TABEL]
{str_tabel}

[KONTEKS GRAFIK/CHART]
{str_gambar}

[PERTANYAAN]
{query}

[INSTRUKSI JAWABAN]
- Jawab dalam Bahasa Indonesia
- Sebutkan sumber informasi (halaman, nama tabel, atau nama grafik)
- Untuk angka finansial, sertakan satuan (Juta/Miliar Rupiah)
- Jika ada ketidaksesuaian antar sumber, sebutkan
"""
        return prompt

# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(BASE_DIR))
    
    from src.retrieval.reranker import RerankerPipeline
    
    retriever_pipeline = RerankerPipeline()
    prompt_builder = PromptBuilder()
    
    print("\n" + "="*50)
    print("M-RAG Prompt Builder Tester")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50)
    
    while True:
        query = input("\nMasukkan Pertanyaan: ")
        if query.lower() in ['keluar', 'exit', 'quit']:
            break
            
        emiten_filter = input("Filter Emiten (contoh: BBCA): ").strip().upper()
        
        filters = {}
        if emiten_filter:
            filters["emiten"] = emiten_filter
            
        print("\nMengambil dokumen & merakit prompt")
        # Ambil Top-5 menggunakan Reranker
        docs = retriever_pipeline.search(query, top_k=5, filters=filters)
        
        if not docs:
            print("Tidak ada dokumen, prompt tidak dapat dibangun.")
            continue
            
        final_prompt = prompt_builder.build_prompt(query, docs)
        
        print("\n" + "="*20 + " HASIL PROMPT FINAL " + "="*20)
        print(final_prompt)
        print("="*60)