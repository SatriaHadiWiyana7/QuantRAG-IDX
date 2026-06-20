"""
Script untuk Dense Retrieval (Vector Similarity Search).
Mengambil konteks (teks, tabel, gambar) yang paling relevan dari ChromaDB
menggunakan pencarian vektor (Cosine Similarity) dengan filter metadata.
"""

import logging
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Setup Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

class DenseRetriever:
    def __init__(self):
        logger.info("Inisialisasi Dense Retriever...")
        
        # 1. Load ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.text_collection = self.chroma_client.get_collection(name="idx_text")
        self.image_collection = self.chroma_client.get_collection(name="idx_image")
        
        # 2. Load Models
        logger.info("Memuat Text Model (MPNet)...")
        self.text_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        
        logger.info("Memuat Image-Text Model (CLIP)...")
        self.clip_model = SentenceTransformer("clip-ViT-L-14")
        
        logger.info("Retriever siap digunakan!")

    def _normalize_l2(self, embeddings):
        """L2 Normalization agar sejalan dengan proses indexing."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def _build_where_clause(self, filters: dict) -> dict:
        """Mengubah dictionary filter menjadi format Where ChromaDB."""
        if not filters:
            return None
            
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            return {key: val}
            
        # Jika ada banyak filter, gunakan $and
        where_conditions = [{k: v} for k, v in filters.items()]
        return {"$and": where_conditions}

    def search_text_and_tables(self, query: str, top_k: int = 5, filters: dict = None):
        """Mencari teks naratif dan tabel finansial yang relevan."""
        # 1. Embed query
        query_emb = self.text_model.encode([query], show_progress_bar=False)
        query_emb_norm = self._normalize_l2(query_emb)
        
        # 2. Setup filter metadata (emiten, tahun, sektor, dll)
        where_clause = self._build_where_clause(filters)
        
        # 3. Search di ChromaDB
        results = self.text_collection.query(
            query_embeddings=query_emb_norm.tolist(),
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)

    def search_images(self, query: str, top_k: int = 3, filters: dict = None):
        """Mencari gambar relevan menggunakan CLIP Text Encoder."""
        # 1. Embed query teks menggunakan CLIP!
        query_emb = self.clip_model.encode([query], show_progress_bar=False)
        query_emb_norm = self._normalize_l2(query_emb)
        
        where_clause = self._build_where_clause(filters)
        
        results = self.image_collection.query(
            query_embeddings=query_emb_norm.tolist(),
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "distances"] # Gambar asli tidak disimpan di dokumen Chroma
        )
        
        return self._format_results(results, is_image=True)

    def _format_results(self, chroma_results, is_image=False):
        """Memformat raw output dari ChromaDB menjadi list of dicts."""
        formatted = []
        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return formatted
            
        ids = chroma_results["ids"][0]
        distances = chroma_results["distances"][0]
        metadatas = chroma_results["metadatas"][0]
        
        documents = chroma_results.get("documents", [[]])[0] if not is_image else ["<IMAGE VECTOR>"] * len(ids)

        for i in range(len(ids)):
            formatted.append({
                "id": ids[i],
                "score": 1.0 - distances[i], # Konversi distance (cosine) ke similarity score
                "content": documents[i],
                "metadata": metadatas[i]
            })
            
        return formatted

# ==========================================
# BLOK INTERAKTIF UNTUK PENGUJIAN MANUAL
# ==========================================
if __name__ == "__main__":
    retriever = DenseRetriever()
    
    print("\n" + "="*50)
    print("M-RAG Dense Retrieval Tester")
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
            
        print("\Mencari Teks & Tabel...")
        text_results = retriever.search_text_and_tables(query, top_k=3, filters=filters)
        
        for i, res in enumerate(text_results):
            mod = res['metadata'].get('modalitas', 'unknown')
            emit = res['metadata'].get('emiten', 'unknown')
            page = res['metadata'].get('halaman', '?')
            print(f"\n[{i+1}] {mod.upper()} | {emit} (Hal {page}) | Score: {res['score']:.4f}")
            # Tampilkan maksimal 300 karakter agar terminal tidak penuh
            print(f"Konten: {res['content'][:300]}...") 
            
        print("\nMencari Gambar/Grafik Terkait...")
        img_results = retriever.search_images(query, top_k=1, filters=filters)
        for i, res in enumerate(img_results):
            emit = res['metadata'].get('emiten', 'unknown')
            img_path = res['metadata'].get('image_path', 'unknown')
            print(f"\n[Img-1] {emit} | Score: {res['score']:.4f}")
            print(f"Path Gambar: {img_path}")
            
    print("Terima kasih telah menguji Retriever!")