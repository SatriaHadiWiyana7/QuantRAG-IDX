"""
Script untuk Sparse Retrieval (Keyword Matching/BM25).
Mengambil konteks yang paling relevan berdasarkan pencocokan kata kunci eksak.
Sangat efektif untuk mencari angka spesifik, nama entitas, atau singkatan.
"""

import string
import logging
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

class SparseRetriever:
    def __init__(self):
        logger.info("Inisialisasi Sparse Retriever (BM25)...")
        
        # Koneksi ke ChromaDB untuk mengambil corpus data
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.text_collection = self.chroma_client.get_collection(name="idx_text")
        
        # Memuat semua dokumen dari Vector Store untuk membangun index BM25
        logger.info("Memuat corpus data untuk indexing BM25 (ini mungkin memakan waktu sebentar)...")
        
        self.ids = []
        self.documents = []
        self.metadatas = []
        
        batch_size = 5000
        offset = 0
        
        # Tarik data secara mencicil (pagination) agar tidak menabrak limit SQLite
        while True:
            batch_data = self.text_collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset
            )
            
            if not batch_data["ids"]:
                break # Berhenti jika sudah tidak ada data lagi
                
            self.ids.extend(batch_data["ids"])
            self.documents.extend(batch_data["documents"])
            self.metadatas.extend(batch_data["metadatas"])
            
            offset += batch_size
            logger.info(f"  ... {len(self.ids)} dokumen berhasil ditarik")
        
        logger.info(f"Total {len(self.documents)} dokumen dimuat. Memulai tokenisasi...")
        tokenized_corpus = [self.tokenize(doc) for doc in self.documents]
        
        logger.info("Membangun model BM25Okapi...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Sparse Retriever siap digunakan!")

    def tokenize(self, text: str) -> list:
        """Tokenizer sederhana untuk mengubah teks menjadi daftar kata."""
        if not text:
            return []
        text = text.lower()
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        """Mencari teks/tabel relevan menggunakan algoritma pencocokan keyword BM25."""
        tokenized_query = self.tokenize(query)
        
        # Hitung skor BM25 untuk semua dokumen terhadap query
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Gabungkan skor dengan index aslinya
        scored_docs = [(i, score) for i, score in enumerate(doc_scores) if score > 0]
        
        # Urutkan berdasarkan skor tertinggi
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_docs:
            meta = self.metadatas[idx]
            
            # Terapkan filter metadata secara manual (karena BM25 berjalan di memori)
            if filters:
                match = True
                for k, v in filters.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
                    
            results.append({
                "id": self.ids[idx],
                "score": score, # Skor BM25 (bukan rentang 0-1 seperti Cosine)
                "content": self.documents[idx],
                "metadata": meta
            })
            
            if len(results) >= top_k:
                break
                
        return results

# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
if __name__ == "__main__":
    retriever = SparseRetriever()
    
    print("\n" + "="*50)
    print("M-RAG Sparse Retrieval (BM25) Tester")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50)
    
    while True:
        query = input("\nMasukkan Kata Kunci: ")
        if query.lower() in ['keluar', 'exit', 'quit']:
            break
            
        emiten_filter = input("Filter Emiten (contoh: BBCA, kosongi jika tidak ada): ").strip().upper()
        
        filters = {}
        if emiten_filter:
            filters["emiten"] = emiten_filter
            
        print("\nMencari pencocokan kata kunci eksak...")
        results = retriever.search(query, top_k=3, filters=filters)
        
        if not results:
            print("Tidak ada dokumen yang mengandung kata kunci tersebut.")
        else:
            for i, res in enumerate(results):
                mod = res['metadata'].get('modalitas', 'unknown')
                emit = res['metadata'].get('emiten', 'unknown')
                page = res['metadata'].get('halaman', '?')
                print(f"\n[{i+1}] {mod.upper()} | {emit} (Hal {page}) | Skor BM25: {res['score']:.4f}")
                print(f"Konten: {res['content'][:300]}...")
                
    print("Terima kasih telah menguji Sparse Retriever!")