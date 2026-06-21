"""
Aplikasi Utama M-RAG Laporan Keuangan IDX (Auto-Theme Edition).
Solusi mutlak untuk UI Bersih, Terang, dengan Auto-Config Streamlit.
"""

import streamlit as st
import sys
import time
import os
from pathlib import Path

# --- 0. AUTO-KONFIGURASI TEMA STREAMLIT (ANTI DARK-MODE) ---
# Skrip ini akan otomatis membuat file konfigurasi resmi Streamlit
# agar aplikasi 100% menggunakan Light Theme secara stabil.
try:
    os.makedirs(".streamlit", exist_ok=True)
    config_path = ".streamlit/config.toml"
    theme_config = """[theme]
base="light"
primaryColor="#2563eb"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f8fafc"
textColor="#0f172a"
font="sans serif"
"""
    # Jika file config belum ada atau bukan light mode, buat ulang
    if not os.path.exists(config_path) or "base=\"light\"" not in open(config_path).read():
        with open(config_path, "w") as f:
            f.write(theme_config)
        st.warning("🔄 Mengkalibrasi tema antarmuka... Silakan segarkan (refresh) browser Anda.")
        st.stop()
except Exception as e:
    pass # Lanjutkan jika ada masalah permission

# --- SETUP JALUR DIREKTORI ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from src.retrieval.reranker import RerankerPipeline
    from src.generation.prompt_builder import PromptBuilder
    from src.generation.answer_generator import AnswerGenerator
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.error("Modul backend tidak ditemukan. Pastikan Anda berada di direktori Riset_UI.")

# --- 1. KONFIGURASI HALAMAN & CUSTOM CSS (HANYA UNTUK ANIMASI & BENTUK) ---
st.set_page_config(page_title="M-RAG Finansial IDX", page_icon="📈", layout="wide")

USER_ICON = "https://api.iconify.design/lucide/user.svg?color=%23334155"
AI_ICON = "https://api.iconify.design/lucide/cpu.svg?color=%232563eb"

st.markdown("""
<style>
    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* BUBBLE CHAT MODERN (Hanya mengatur bentuk dan border) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f1f5f9; border-radius: 12px; padding: 1.5rem; border: 1px solid #e2e8f0;
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff; border-left: 4px solid #2563eb; border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    /* FIX TABEL MARKDOWN */
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; background-color: #ffffff; font-size: 0.95rem; }
    th { background-color: #f8fafc; font-weight: 600; text-align: left; padding: 12px; border-bottom: 2px solid #cbd5e1; }
    td { padding: 12px; border-bottom: 1px solid #e2e8f0; }
    
    /* Hapus background Avatar SVG */
    [data-testid="stChatMessageAvatar"] { background-color: transparent !important; }
    .stChatInputContainer { padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNGSI UTILITAS VISUAL ---
def render_visuals(images_list: list):
    if not images_list: return
    st.markdown("<br><strong>📊 Visualisasi Terkait:</strong>", unsafe_allow_html=True)
    cols = st.columns(min(len(images_list), 3))
    for idx, img_path in enumerate(images_list):
        with cols[idx % 3]:
            if os.path.exists(img_path): st.image(img_path, use_container_width=True)

def render_citations(docs: list):
    if not docs: return
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Jejak Audit & Transparansi Konteks"):
        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            st.markdown(f"**[{i+1}] {meta.get('modalitas', '').capitalize()} | {meta.get('emiten', '')} (Halaman {meta.get('halaman', '?')})**")
            st.caption(f"_{doc.get('content', '')[:200]}..._")

# --- 3. INISIALISASI BACKEND & STATE ---
@st.cache_resource
def load_backend():
    if BACKEND_AVAILABLE:
        return {"retriever": RerankerPipeline(), "prompt_builder": PromptBuilder(), "generator": AnswerGenerator()}
    return None

backend = load_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    st.markdown("---")
    emiten_input = st.text_input("Filter Emiten:", placeholder="Contoh: BBCA").upper()
    lang_toggle = st.radio("Bahasa Respons:", ["Indonesia", "English"], horizontal=True)
    st.session_state.language = "ID" if lang_toggle == "Indonesia" else "EN"
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Mulai Sesi Baru", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. ANTARMUKA UTAMA ---
st.markdown("## 📈 M-RAG Laporan Keuangan IDX")
st.markdown("Asisten analitik presisi tinggi. Mendukung pencarian teks naratif, ekstraksi tabel, dan analisis visual.")
st.markdown("<br>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    current_icon = USER_ICON if msg["role"] == "user" else AI_ICON
    with st.chat_message(msg["role"], avatar=current_icon):
        st.markdown(msg["content"])
        render_visuals(msg.get("images", []))

# --- 6. LOGIKA PEMROSESAN CHAT ---
if prompt := st.chat_input("Tanyakan wawasan finansial... (contoh: Berapa total aset 2023?)"):
    
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "images": []})

    with st.chat_message("assistant", avatar=AI_ICON):
        msg_placeholder = st.empty()
        images_to_display = []
        
        if not BACKEND_AVAILABLE:
            msg_placeholder.markdown("Sistem backend belum terhubung.")
        else:
            with st.spinner("🔍 Memindai dokumen..."):
                filters = {"emiten": emiten_input} if emiten_input else None
                docs = backend["retriever"].search(prompt, top_k=15, filters=filters)
                
            if not docs:
                msg_placeholder.markdown("Maaf, informasi tersebut tidak ditemukan.")
            else:
                for doc in docs:
                    meta = doc.get("metadata", {})
                    if meta.get("modalitas") == "gambar" and "image_path" in meta:
                        img_path = Path(BASE_DIR) / meta["image_path"]
                        if img_path.exists() and str(img_path) not in images_to_display:
                            images_to_display.append(str(img_path))

                with st.spinner("🧠 Menyusun analisis..."):
                    raw_prompt = backend["prompt_builder"].build_prompt(prompt, docs)
                    
                    # INSTRUKSI TABEL DIPERKETAT AGAR TIDAK TERPOTONG
                    lang_instruction = (
                    "Jawab dalam Bahasa Indonesia layaknya analis keuangan senior: natural dan informatif. "
                    "Jika Anda menyebutkan angka, Anda WAJIB merangkumnya ke dalam tabel Markdown berisikan tepat 4 kolom, yaitu: "
                    "Kolom 1 untuk Entitas, Kolom 2 untuk Metrik, Kolom 3 untuk Nilai, dan Kolom 4 untuk Sumber (Dokumen & Hal). "
                    "Tuntaskan jawaban Anda sampai akhir, jangan ada kalimat atau tabel yang terpotong."
                    ) if st.session_state.language == "ID" else (
                        "Answer in English as a senior financial analyst: natural and informative. "
                        "If you mention numbers, you MUST summarize them in a Markdown table containing exactly 4 columns: "
                        "Column 1 for Entity, Column 2 for Metric, Column 3 for Value, and Column 4 for Source (Doc & Page). "
                        "Finish your response completely, do not cut off any sentences or tables."
                    )
                    final_prompt = raw_prompt.replace("Jawab dalam Bahasa Indonesia", lang_instruction)
                    response = backend["generator"].generate_answer(final_prompt)
                
                # Render seluruh hasil secara instan
                msg_placeholder.markdown(response)
                render_visuals(images_to_display)
                render_citations(docs)

        if BACKEND_AVAILABLE:
            st.session_state.messages.append({
                "role": "assistant", "content": response, "images": images_to_display
            })