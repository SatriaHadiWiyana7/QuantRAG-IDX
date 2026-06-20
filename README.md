# QuantRAG-IDX 📊🤖

QuantRAG-IDX adalah sistem **Multimodal Retrieval-Augmented Generation (M-RAG)** mutakhir yang dirancang khusus untuk mengekstrak, menganalisis, dan melakukan tanya-jawab cerdas berbasis dokumen laporan keuangan (*Annual Report*) perusahaan publik yang terdaftar di Bursa Efek Indonesia (IDX).

Sistem ini tidak hanya membaca teks naratif biasa, melainkan mengintegrasikan tiga modalitas krusial secara hibrida: **teks finansial**, **tabel keuangan terstruktur**, dan **grafik/chart performa bisnis**.

---

## 🚀 Fitur Utama & Keunggulan
- **Multimodal Pipeline:** Pemrosesan simultan untuk teks (naratif), tabel (neraca, laba/rugi), dan visual (grafik batang, garis, lingkaran) menggunakan Vision-Language Models (VLM).
- **Robust Table Serialization:** Mengonversi tabel numerik kompleks menjadi representasi tekstual terstruktur guna mempertahankan konteks hubungan antarbaris dan kolom.
- **Hybrid Multi-Index Retrieval:** Menggabungkan pencarian semantik (*dense retrieval*) berbasis kedekatan vektor dengan pencarian kata kunci (*sparse retrieval* BM25) menggunakan metode *Reciprocal Rank Fusion (RRF)*.
- **Financial-Domain Guardrails:** Verifikasi nilai numerik dan kepatuhan unit finansial secara ketat demi meminimalkan risiko halusinasi informasi pada LLM[cite: 1].

---

## 🛠️ Stack Teknologi
- **Bahasa:** Python 3.10+[cite: 1]
- **PDF Extraction:** `pymupdf` (Fitz), `camelot-py`, `pdfplumber`[cite: 1]
- **Vision Models:** InternVL2 / LLaVA (via Ollama) / GPT-4V[cite: 1]
- **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (Teks/Tabel) & CLIP (Gambar)[cite: 1]
- **Vector Store:** ChromaDB / Qdrant[cite: 1]
- **LLM Generator:** Llama-3-8B-Instruct / Mistral-7B / GPT-4o-mini[cite: 1]
- **Framework Evaluasi:** RAGAS, BERTScore, & Custom Financial Metrics[cite: 1]

---

## 📂 Struktur Proyek
```text
QuantRAG-IDX/
├── data/               # Dataset PDF raw, hasil processed, dan ground-truth QA (Diabaikan oleh Git)
├── src/                # Kode sumber utama aplikasi
│   ├── ingestion/      # Ekstraksi PDF, tabel, gambar, dan pembuatan caption grafik
│   ├── indexing/       # Pembuatan embedding teks, tabel, dan gambar ke Vector Store
│   ├── retrieval/      # Strategi Dense, Sparse, Hybrid, dan Cross-Encoder Reranking
│   ├── generation/     # Prompt builder, integrasi LLM, dan mekanisme anti-halusinasi
│   ├── evaluation/     # Framework pengujian metrik RAGAS dan akurasi keuangan
│   └── utils/          # Logger, konfigurasi hyperparameter, dan validator data
├── experiments/        # Dokumentasi konfigurasi eksperimen, notebook eksplorasi, dan hasil studi ablasi
├── tests/              # Unit testing komponen kritis pipeline
└── scripts/            # Skrip otomatisasi eksekusi pipeline end-to-end