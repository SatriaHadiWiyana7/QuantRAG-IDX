"""
Script untuk LLM Generation menggunakan Google Gemini API.
Menerima prompt yang sudah dirakit oleh PromptBuilder, lalu mengirimkannya ke Gemini
untuk menghasilkan jawaban akhir yang akurat dan berbasis konteks.
"""

import os
import sys
import logging
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Daftarkan folder root (Riset_UI) ke sistem Python
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Muat environment variables dari file .env
load_dotenv(BASE_DIR / ".env")

from src.generation.prompt_builder import PromptBuilder
from src.retrieval.reranker import RerankerPipeline

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        logger.info("Inisialisasi Answer Generator (Google Gemini API)...")
        
        # Mengambil API Key dari file .env
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "masukkan_api_key_gemini_anda_di_sini":
            logger.error("GEMINI_API_KEY tidak valid atau belum diisi di file .env!")
            sys.exit(1)
            
        genai.configure(api_key=api_key)
        
        # Menggunakan Gemini 2.5 Flash 
        self.model_name = 'gemini-2.5-flash'
        logger.info(f"LLM Client siap menggunakan model: {self.model_name}")
        
        # Inisialisasi model dengan instruksi ketat
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
            )
        )

    def generate_answer(self, prompt: str) -> str:
        """
        Mengirimkan prompt raksasa (Sistem + Konteks + Query) ke Gemini.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gagal menghasilkan jawaban dari Gemini: {e}")
            return "Maaf, terjadi kesalahan saat menghubungi model AI Gemini."

# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
if __name__ == "__main__":
    retriever_pipeline = RerankerPipeline()
    prompt_builder = PromptBuilder()
    generator = AnswerGenerator()
    
    print("\n" + "="*60)
    print("M-RAG End-to-End Pipeline Tester")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*60)
    
    while True:
        query = input("\nMasukkan Pertanyaan: ")
        if query.lower() in ['keluar', 'exit', 'quit']:
            break
            
        emiten_filter = input("Filter Emiten (contoh: BBCA): ").strip().upper()
        
        filters = {}
        if emiten_filter:
            filters["emiten"] = emiten_filter
            
        print("\nMengambil dan mererank dokumen...")
        docs = retriever_pipeline.search(query, top_k=15, filters=filters)
        
        if not docs:
            print("Tidak ada dokumen yang ditemukan di database.")
            continue
            
        print("Membangun Prompt dari Konteks...")
        prompt = prompt_builder.build_prompt(query, docs)
        
        print("Meminta Gemini Berpikir dan Menjawab Pertanyaan...")
        jawaban = generator.generate_answer(prompt)
        
        print("\n" + "="*20 + " JAWABAN M-RAG " + "="*20)
        print(jawaban)
        print("="*55)