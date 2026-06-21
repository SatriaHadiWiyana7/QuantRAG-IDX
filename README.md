# 📈 M-RAG Laporan Keuangan IDX

Proyek ini merupakan implementasi tingkat lanjut dari **Multimodal Retrieval-Augmented Generation (M-RAG)** yang dirancang khusus untuk menganalisis, mengekstrak, dan menjawab pertanyaan berdasarkan Laporan Tahunan (Annual Report) perusahaan publik di Bursa Efek Indonesia (IDX). 

Sistem ini mampu memproses berbagai modalitas data dari dokumen PDF finansial, termasuk teks naratif yang panjang, tabel tanpa batas (*borderless tables*), dan elemen visual (grafik/chart).

---

## 🏗️ Arsitektur Sistem

Proyek ini dibagi menjadi empat fase utama (Pipeline End-to-End):

1. **Ingestion (Pemrosesan Data Mentah)**
   - Mengekstrak teks naratif dan membaginya menjadi *chunks* yang dapat dikelola.
   - Mengekstrak struktur tabel finansial secara presisi (menggunakan *Camelot*).
   - Mengekstrak aset visual (grafik/chart) dan memberikan *caption* tekstual menggunakan model *Vision*.
2. **Indexing (Penyimpanan Vektor)**
   - Mengonversi data multimodal menjadi representasi vektor numerik berdimensi tinggi.
   - Menyimpannya ke dalam *Vector Database* lokal (**ChromaDB**) dengan pemisahan koleksi `idx_text` dan `idx_image`.
3. **Retrieval (Pencarian & Pengurutan Ulang)**
   - **Hybrid Retrieval:** Menggabungkan *Dense Retrieval* (Pencarian Semantik Berbasis Vektor) dan *Sparse Retrieval* (Pencarian Kata Kunci Eksak / BM25).
   - **Reciprocal Rank Fusion (RRF):** Menggabungkan skor dari sistem Dense dan Sparse.
   - **Cross-Encoder Reranking:** Menilai ulang (rerank) kandidat dokumen (Top 75) untuk menyingkirkan *noise* bahasa asing dan memastikan presisi maksimal dari konteks yang dikembalikan.
4. **Generation (Pembangkitan Jawaban LLM)**
   - Merakit *prompt* yang ketat (anti-halusinasi) dari gabungan teks, tabel, dan caption.
   - Meminta *Large Language Model* (LLM) untuk menjawab dan menyertakan sitasi halaman secara eksplisit.

---

## 🧠 Spesifikasi Model

Sistem ini memanfaatkan kombinasi model *Open-Source* dan *Cloud API* berkinerja tinggi:

- **LLM Generator:** `gemini-2.5-flash` (Google Gemini API) - *Dipilih karena kemampuannya dalam penalaran konteks panjang dan merajut teks tabel yang terfragmentasi.*
- **Text Embedding:** `paraphrase-multilingual-mpnet-base-v2` (SentenceTransformers) - *Optimal untuk memahami bahasa Indonesia dan terminologi finansial dwibahasa.*
- **Image/Vision Embedding:** `clip-ViT-L-14` (OpenAI CLIP) - *Digunakan untuk ekstraksi fitur visual dan Cross-Modal Linking antara gambar dan teks.*
- **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` - *Berperan vital dalam memfilter dokumen dari jebakan kosakata (Vocabulary Mismatch).*
- **Sparse Retriever:** `rank_bm25` (BM25Okapi) - *Pencocokan kata kunci eksak untuk pencarian angka dan metrik finansial (exact needle-in-a-haystack).*

---

## 📂 Struktur Direktori Utama

```text
Riset_UI/
│
├── data/
│   ├── raw/                 # File PDF mentah (contoh: BBCA_2023_annual_report.pdf)
│   ├── processed/           # Hasil chunking teks, JSON tabel, dan ekstraksi gambar
│   └── vector_store/        # Basis data vektor lokal (ChromaDB)
│
├── src/
│   ├── ingestion/           # Pipeline ekstraksi (PDF, Tabel, Gambar, Metadata)
│   ├── indexing/            # text_embedder.py, table_embedder.py, image_embedder.py
│   ├── retrieval/           # dense_retriever.py, sparse_retriever.py, hybrid_retriever.py, reranker.py
│   └── generation/          # prompt_builder.py, answer_generator.py
│
├── .env                     # Konfigurasi Environment Variables (API Keys)
└── README.md                # Dokumentasi proyek