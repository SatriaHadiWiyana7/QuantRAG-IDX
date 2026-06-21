"""
Script untuk Cross-Encoder Reranking.
Mengambil kandidat dokumen dari Hybrid Retriever, lalu menilainya ulang (rerank)
secara komprehensif menggunakan model Cross-Encoder untuk mendapatkan presisi maksimal.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict
from sentence_transformers import CrossEncoder

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.retrieval.hybrid_retriever import HybridRetriever

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RerankerPipeline:
    def __init__(self):
        logger.info("Inisialisasi Reranker Pipeline")
        self.hybrid_retriever = HybridRetriever()
        
        # Menggunakan model Cross-Encoder sesuai panduan
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Memuat Cross-Encoder Model ({model_name})...")
        self.reranker = CrossEncoder(model_name, max_length=512)
        
        logger.info("Reranker Pipeline siap digunakan!")

    def search(self, query: str, top_k: int = 5, filters: dict = None) -> List[Dict]:
        """
        Melakukan pencarian 2-Tahap:
        Tahap 1: Recall (Ambil Top-50 dari Hybrid/RRF)
        Tahap 2: Precision (Rerank menggunakan Cross-Encoder jadi Top-5)
        """
        # Ambil kandidat kasar yang banyak
        candidate_k = max(75, top_k * 10)
        logger.info(f"TAHAP 1: Mengambil {candidate_k} kandidat dari Hybrid Retriever")
        candidates = self.hybrid_retriever.search(query, top_k=candidate_k, filters=filters)
        
        if not candidates:
            return []

        # Siapkan pasangan [Query, Dokumen] untuk dibaca sekaligus oleh AI
        logger.info("TAHAP 2: Melakukan Reranking menggunakan Cross-Encoder")
        pairs = []
        for doc in candidates:
            pairs.append([query, doc.get("content", "")])

        # Hitung skor relevansi sejati
        # Cross-Encoder akan memprediksi seberapa logis dokumen tersebut menjawab query
        scores = self.reranker.predict(pairs)

        # Gabungkan skor prediksi dengan data kandidat aslinya
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # Urutkan ulang berdasarkan skor Reranker (dari yang paling relevan ke yang tidak)
        reranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Kembalikan hanya top_k yang diminta
        return reranked_results[:top_k]

# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
if __name__ == "__main__":
    pipeline = RerankerPipeline()
    
    print("\n" + "="*50)
    print("M-RAG Reranker Pipeline Tester")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50)
    
    while True:
        query = input("\nMasukkan Pertanyaan: ")
        if query.lower() in ['keluar', 'exit', 'quit']:
            break
            
        emiten_filter = input("Filter Emiten (contoh: BBCA, kosongi jika tidak ada): ").strip().upper()
        
        filters = {}
        if emiten_filter:
            filters["emiten"] = emiten_filter
            
        print("\nMencari dan mengurutkan ulang (Reranking) dokumen...")
        results = pipeline.search(query, top_k=3, filters=filters)
        
        if not results:
            print("Tidak ada dokumen yang ditemukan.")
        else:
            for i, res in enumerate(results):
                mod = res['metadata'].get('modalitas', 'unknown')
                emit = res['metadata'].get('emiten', 'unknown')
                page = res['metadata'].get('halaman', '?')
                
                print(f"\n[{i+1}] {mod.upper()} | {emit} (Hal {page}) | Rerank Score: {res['rerank_score']:.4f}")
                print(f"Skor Asal RRF: {res.get('rrf_score', 0):.4f}")
                print(f"Konten: {res['content'][:300]}...")
                
    print("Terima kasih telah menguji Reranker Pipeline!")