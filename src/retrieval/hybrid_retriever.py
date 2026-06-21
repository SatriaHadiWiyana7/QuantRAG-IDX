"""
Script untuk Hybrid Retrieval.
Menggabungkan hasil dari Dense Retrieval (Vector) dan Sparse Retrieval (BM25)
menggunakan algoritma Reciprocal Rank Fusion (RRF).
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        logger.info("Inisialisasi Hybrid Retriever...")
        # Memuat kedua retriever yang sudah kita buat sebelumnya
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        logger.info("Hybrid Retriever siap digunakan!")

    def _compute_rrf(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Menghitung Reciprocal Rank Fusion (RRF) dari dua list hasil pencarian.
        Formula: RRF_score = 1 / (k + rank)
        """
        rrf_scores = {}
        docs_dict = {}

        # Proses skor peringkat dari Dense Results
        for rank, res in enumerate(dense_results):
            doc_id = res['id']
            docs_dict[doc_id] = res
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Proses skor peringkat dari Sparse Results (BM25)
        for rank, res in enumerate(sparse_results):
            doc_id = res['id']
            if doc_id not in docs_dict:
                docs_dict[doc_id] = res
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Urutkan dokumen berdasarkan Total RRF Score tertinggi
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs:
            doc = docs_dict[doc_id].copy()
            doc['rrf_score'] = score
            final_results.append(doc)

        return final_results

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        """
        Melakukan pencarian hybrid (Dense + Sparse) dan menggabungkannya dengan RRF.
        """
        # Kita ambil jumlah kandidat yang lebih banyak (2x lipat dari top_k) 
        candidate_k = max(10, top_k * 2)
        
        logger.info("Menjalankan Dense Search")
        dense_res = self.dense_retriever.search_text_and_tables(query, top_k=candidate_k, filters=filters)
        
        logger.info("Menjalankan Sparse Search (BM25)")
        sparse_res = self.sparse_retriever.search(query, top_k=candidate_k, filters=filters)
        
        logger.info("Menjalankan Reciprocal Rank Fusion (RRF)")
        hybrid_res = self._compute_rrf(dense_res, sparse_res)
        
        # Kembalikan hanya top_k yang diminta pengguna
        return hybrid_res[:top_k]

# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    print("\n" + "="*50)
    print("M-RAG Hybrid Retrieval (RRF) Tester")
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
            
        print("\nMencari menggunakan kombinasi Semantik & BM25")
        results = retriever.search(query, top_k=5, filters=filters)
        
        if not results:
            print("Tidak ada dokumen yang ditemukan.")
        else:
            for i, res in enumerate(results):
                mod = res['metadata'].get('modalitas', 'unknown')
                emit = res['metadata'].get('emiten', 'unknown')
                page = res['metadata'].get('halaman', '?')
                
                print(f"\n[{i+1}] {mod.upper()} | {emit} (Hal {page}) | Skor RRF: {res['rrf_score']:.4f}")
                # Tampilkan juga asal skor aslinya jika ingin inspeksi (opsional)
                score_asal = res.get('score', 0)
                print(f"Skor Asal (Vector/BM25): {score_asal:.4f}")
                print(f"Konten: {res['content'][:300]}...")
                
    print("Terima kasih telah menguji Hybrid Retriever!")