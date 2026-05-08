import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Konfigurasi Halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Webtoon",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: bold;
        color: #1a1a2e; text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .subtitle {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px;
        color: white; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .result-positive {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 1.5rem; border-radius: 10px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .result-negative {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; padding: 1.5rem; border-radius: 10px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .info-box {
        background: #f0f2f6; padding: 1rem;
        border-radius: 8px; border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .step-box {
        background: white; padding: 0.8rem 1rem;
        border-radius: 8px; margin: 0.3rem 0;
        border: 1px solid #e0e0e0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('model_SVM_best.pkl', 'rb') as f:
            saved = pickle.load(f)
        return saved['model'], saved['tfidf']
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_nltk():
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        sw = set(stopwords.words('indonesian'))
        # Hapus kata negasi dari stopwords
        negasi = {'tidak','tak','bukan','bukanlah','kurang','jangan',
                  'belum','tanpa','enggak','nggak','ngga'}
        sw = sw - negasi
        custom_sw = {
            'yg','ny','nya','deh','sih','nih','aja','ajah',
            'udah','udh','dah','bgt','bngtt','bngt','dong','loh','lho','lah',
            'yaa','yaaa','yap','yep','webtoon','app','aplikasi',
            'line','play','store','google','naver',
            'skrg','skrang','barusan','gue','gw','ane','lo','lu',
        }
        sw = sw.union(custom_sw)

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer, sw, word_tokenize
    except Exception as e:
        return None, None, None

def preprocess(text, stemmer, stop_words, tokenize):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Normalisasi slang
    slang = {'ga':'tidak','gak':'tidak','gk':'tidak','ngga':'tidak',
             'nggak':'tidak','tdk':'tidak','gabisa':'tidak bisa',
             'gaada':'tidak ada','bgt':'banget','bngt':'banget',
             'kalo':'kalau','krn':'karena','yg':'yang','udh':'sudah',
             'blm':'belum','tp':'tapi','tpi':'tapi'}
    words = text.split()
    text = ' '.join([slang.get(w, w) for w in words])

    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ── Data hasil penelitian ──────────────────────────────────────────────────────
HASIL_PENELITIAN = [
    {"Kode":"EXP-01","Algoritma":"SVM","N-Gram":"Unigram (1,1)","Accuracy":0.9130,"Precision":0.9095,"Recall":0.9130,"F1-Score":0.9100,"Waktu (s)":1.48},
    {"Kode":"EXP-02","Algoritma":"SVM","N-Gram":"Bigram (2,2)","Accuracy":0.8687,"Precision":0.8589,"Recall":0.8687,"F1-Score":0.8504,"Waktu (s)":1.31},
    {"Kode":"EXP-03","Algoritma":"SVM","N-Gram":"Trigram (3,3)","Accuracy":0.8306,"Precision":0.8404,"Recall":0.8306,"F1-Score":0.7659,"Waktu (s)":0.87},
    {"Kode":"EXP-04","Algoritma":"SVM","N-Gram":"Unigram+Bigram (1,2) ★","Accuracy":0.9153,"Precision":0.9124,"Recall":0.9153,"F1-Score":0.9132,"Waktu (s)":2.30},
    {"Kode":"EXP-05","Algoritma":"SVM","N-Gram":"Unigram+Bigram+Trigram (1,3)","Accuracy":0.9130,"Precision":0.9097,"Recall":0.9130,"F1-Score":0.9104,"Waktu (s)":6.91},
    {"Kode":"EXP-06","Algoritma":"Random Forest","N-Gram":"Unigram (1,1)","Accuracy":0.8889,"Precision":0.8819,"Recall":0.8889,"F1-Score":0.8801,"Waktu (s)":7.55},
    {"Kode":"EXP-07","Algoritma":"Random Forest","N-Gram":"Bigram (2,2)","Accuracy":0.8640,"Precision":0.8522,"Recall":0.8640,"F1-Score":0.8451,"Waktu (s)":8.00},
    {"Kode":"EXP-08","Algoritma":"Random Forest","N-Gram":"Trigram (3,3)","Accuracy":0.8283,"Precision":0.8105,"Recall":0.8283,"F1-Score":0.7668,"Waktu (s)":9.15},
    {"Kode":"EXP-09","Algoritma":"Random Forest","N-Gram":"Unigram+Bigram (1,2) ★","Accuracy":0.9005,"Precision":0.8955,"Recall":0.9005,"F1-Score":0.8936,"Waktu (s)":5.58},
    {"Kode":"EXP-10","Algoritma":"Random Forest","N-Gram":"Unigram+Bigram+Trigram (1,3)","Accuracy":0.8904,"Precision":0.8842,"Recall":0.8904,"F1-Score":0.8806,"Waktu (s)":4.80},
    {"Kode":"EXP-11","Algoritma":"XGBoost","N-Gram":"Unigram (1,1) ★","Accuracy":0.8858,"Precision":0.9078,"Recall":0.8858,"F1-Score":0.8922,"Waktu (s)":2.68},
    {"Kode":"EXP-12","Algoritma":"XGBoost","N-Gram":"Bigram (2,2)","Accuracy":0.8454,"Precision":0.8345,"Recall":0.8454,"F1-Score":0.8383,"Waktu (s)":4.50},
    {"Kode":"EXP-13","Algoritma":"XGBoost","N-Gram":"Trigram (3,3)","Accuracy":0.8205,"Precision":0.7720,"Recall":0.8205,"F1-Score":0.7487,"Waktu (s)":2.66},
    {"Kode":"EXP-14","Algoritma":"XGBoost","N-Gram":"Unigram+Bigram (1,2)","Accuracy":0.8788,"Precision":0.9025,"Recall":0.8788,"F1-Score":0.8858,"Waktu (s)":6.91},
    {"Kode":"EXP-15","Algoritma":"XGBoost","N-Gram":"Unigram+Bigram+Trigram (1,3)","Accuracy":0.8749,"Precision":0.8960,"Recall":0.8749,"F1-Score":0.8816,"Waktu (s)":8.76},
]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Webtoon_App_Logo.png/240px-Webtoon_App_Logo.png", width=80)
    st.markdown("### 📚 Analisis Sentimen Webtoon")
    st.markdown("---")
    menu = st.radio("Navigasi", [
        "🏠 Beranda",
        "🔍 Prediksi Sentimen",
        "📊 Hasil Penelitian"
    ])
    st.markdown("---")
    st.markdown("**Model Terbaik:**")
    st.markdown("✅ SVM + Unigram+Bigram")
    st.markdown("**Akurasi:** 91.53%")
    st.markdown("**F1-Score:** 91.32%")
    st.markdown("---")
    st.markdown("**Peneliti:**")
    st.markdown("Calvin Jhon")
    st.markdown("Sistem Informasi · 2025")

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown('<div class="main-title">📚 Sistem Analisis Sentimen Ulasan Aplikasi Webtoon</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Implementasi Model SVM dengan Konfigurasi TF-IDF + N-Gram Terbaik</div>', unsafe_allow_html=True)

    # Metrik utama
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">91.53%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">91.24%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">91.53%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">91.32%</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 📋 Tentang Penelitian")
        st.markdown("""
        <div class="info-box">
        Penelitian ini menganalisis pengaruh konfigurasi n-gram pada representasi
        TF-IDF terhadap performa klasifikasi sentimen ulasan aplikasi Webtoon
        di Google Play Store.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Rumusan Masalah:**")
        st.markdown("1. Bagaimana pengaruh konfigurasi n-gram terhadap performa TF-IDF?")
        st.markdown("2. Bagaimana perbandingan performa SVM, Random Forest, dan XGBoost?")
        st.markdown("3. Kombinasi mana yang menghasilkan performa terbaik?")

        st.markdown("**Metodologi:**")
        cols = st.columns(3)
        with cols[0]:
            st.info("**Dataset**\n\n10.000 ulasan\nGoogle Play Store")
        with cols[1]:
            st.info("**Algoritma**\n\nSVM · RF\nXGBoost")
        with cols[2]:
            st.info("**N-Gram**\n\n5 konfigurasi\n15 eksperimen")

    with col_right:
        st.markdown("### ⚙️ Detail Model Terbaik")
        st.success("**SVM + Unigram+Bigram (1,2)**")

        st.markdown("**Pipeline Preprocessing:**")
        steps = [
            "1️⃣ Case Folding — ubah ke huruf kecil",
            "2️⃣ Normalisasi — kurangi karakter berulang",
            "3️⃣ Cleaning — hapus URL, emoji, angka",
            "4️⃣ Tokenisasi — pecah menjadi token",
            "5️⃣ Stopword Removal — hapus kata tidak bermakna",
            "6️⃣ Stemming — reduksi ke kata dasar (PySastrawi)",
            "7️⃣ TF-IDF + Bigram — ekstraksi fitur",
            "8️⃣ SVM Linear — klasifikasi sentimen",
        ]
        for step in steps:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Ringkasan Temuan Penelitian")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success("**Temuan 1**\n\nKonfigurasi Unigram+Bigram (1,2) secara konsisten menghasilkan performa terbaik pada SVM dan Random Forest karena mampu menangkap konteks frasa.")
    with col_b:
        st.info("**Temuan 2**\n\nSVM unggul secara keseluruhan (F1 91,32%). XGBoost lebih seimbang dalam mendeteksi kelas negatif berkat parameter scale_pos_weight.")
    with col_c:
        st.warning("**Temuan 3**\n\nTrigram murni (3,3) menghasilkan performa paling rendah di semua algoritma akibat sparsity tinggi pada data ulasan pendek.")
# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI SENTIMEN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 Prediksi Sentimen":
    st.markdown("## 🔍 Prediksi Sentimen Ulasan Webtoon")
    st.markdown("Masukkan ulasan pengguna aplikasi Webtoon untuk diklasifikasikan sentimennya.")
 
    model, tfidf = load_model()
    stemmer, stop_words, tokenize = load_nltk()
 
    if model is None:
        st.error("⚠️ Model tidak ditemukan. Pastikan file `model_SVM_best.pkl` tersedia.")
        st.stop()
 
    if stemmer is None:
        st.error("⚠️ Library preprocessing tidak berhasil dimuat.")
        st.stop()
 
    # Input
    st.markdown("### ✍️ Masukkan Ulasan")
    contoh_options = {
        "Tulis sendiri...": "",
        "Contoh Positif 1": "aplikasi ini bagus banget banyak komik seru yang bisa dibaca gratis",
        "Contoh Positif 2": "webtoon sangat seru ceritanya menarik dan gambarnya bagus sekali",
        "Contoh Negatif 1": "aplikasi sering error dan lambat sangat mengecewakan",
        "Contoh Negatif 2": "tidak bisa login sudah lama belum ada perbaikan dari developer",
    }
 
    pilihan = st.selectbox("Atau pilih contoh:", list(contoh_options.keys()))
    default_text = contoh_options[pilihan]
 
    user_input = st.text_area(
        "Teks ulasan:",
        value=default_text,
        height=120,
        placeholder="Contoh: aplikasi ini sangat bagus dan banyak komik seru..."
    )
 
    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🔮 Prediksi", type="primary", use_container_width=True)
 
    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Memproses..."):
                clean = preprocess(user_input, stemmer, stop_words, tokenize)
                vec   = tfidf.transform([clean])
                pred  = model.predict(vec)[0]
 
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")
 
            col_res, col_detail = st.columns([1, 1.5])
 
            with col_res:
                if pred == 1:
                    st.markdown('<div class="result-positive">✅ POSITIF</div>', unsafe_allow_html=True)
                    st.success("Ulasan ini mengekspresikan **sentimen positif** — pengguna merasa puas dengan aplikasi Webtoon.")
                else:
                    st.markdown('<div class="result-negative">❌ NEGATIF</div>', unsafe_allow_html=True)
                    st.error("Ulasan ini mengekspresikan **sentimen negatif** — pengguna merasa tidak puas atau kecewa.")
 
            with col_detail:
                st.markdown("**Detail Preprocessing:**")
                st.markdown(f"**Teks asli:** {user_input[:200]}{'...' if len(user_input) > 200 else ''}")
                st.markdown(f"**Setelah preprocessing:**")
                st.code(clean if clean else "(teks kosong setelah preprocessing)")
                st.markdown(f"**Jumlah token:** {len(clean.split()) if clean else 0}")
                st.markdown(f"**Model:** SVM + Unigram+Bigram (1,2)")
 
    st.markdown("---")
    st.markdown("### 📋 Prediksi Banyak Ulasan (Upload CSV)")
    st.markdown("Upload file CSV dengan kolom `content` berisi ulasan-ulasan yang ingin diprediksi.")
 
    uploaded = st.file_uploader("Pilih file CSV", type=['csv'])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write("**Preview data:**")
        st.dataframe(df_upload.head())
 
        if 'content' not in df_upload.columns:
            st.error("❌ File CSV harus memiliki kolom bernama `content`!")
        else:
            if st.button("🚀 Prediksi Semua", type="primary"):
                with st.spinner(f"Memproses {len(df_upload)} ulasan..."):
                    df_upload['clean_text'] = df_upload['content'].apply(
                        lambda x: preprocess(str(x), stemmer, stop_words, tokenize)
                    )
                    vecs = tfidf.transform(df_upload['clean_text'])
                    df_upload['Prediksi'] = model.predict(vecs)
                    df_upload['Sentimen'] = df_upload['Prediksi'].map({1: '✅ Positif', 0: '❌ Negatif'})
 
                st.success(f"✅ Selesai! {len(df_upload)} ulasan diproses.")
 
                col_tbl, col_chart = st.columns([1.5, 1])
                with col_tbl:
                    st.dataframe(df_upload[['content', 'Sentimen']].head(20))
                with col_chart:
                    counts = df_upload['Sentimen'].value_counts()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    colors = ['#38ef7d' if '✅' in x else '#f5576c' for x in counts.index]
                    ax.pie(counts.values, labels=['Positif', 'Negatif'],
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title('Distribusi Sentimen')
                    st.pyplot(fig)
                    plt.close()
 
                csv_out = df_upload[['content', 'Sentimen']].to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil CSV", csv_out,
                                   'hasil_prediksi.csv', 'text/csv')

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — HASIL PENELITIAN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Hasil Penelitian":
    st.markdown("## 📊 Hasil Penelitian — 15 Eksperimen")
    st.markdown("Perbandingan performa seluruh kombinasi konfigurasi n-gram dan algoritma klasifikasi.")

    df_hasil = pd.DataFrame(HASIL_PENELITIAN)

    # Filter
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_algo = st.multiselect(
            "Filter Algoritma:",
            options=["SVM", "Random Forest", "XGBoost"],
            default=["SVM", "Random Forest", "XGBoost"]
        )
    with col_f2:
        sort_by = st.selectbox("Urutkan berdasarkan:", ["F1-Score", "Accuracy", "Precision", "Recall"])

    df_filtered = df_hasil[df_hasil['Algoritma'].isin(filter_algo)]
    df_filtered = df_filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)
    df_filtered.index += 1

    st.dataframe(
        df_filtered[['Kode','Algoritma','N-Gram','Accuracy','Precision','Recall','F1-Score','Waktu (s)']],
        use_container_width=True,
        height=420
    )

    st.markdown("★ = Model terbaik per algoritma")
    st.markdown("---")

    # Bar chart
    st.markdown("### 📈 Perbandingan F1-Score")
    ngram_order = ['Unigram (1,1)','Bigram (2,2)','Trigram (3,3)',
                   'Unigram+Bigram (1,2) ★','Unigram+Bigram+Trigram (1,3)']
    ngram_labels = ['Unigram\n(1,1)','Bigram\n(2,2)','Trigram\n(3,3)',
                    'Uni+Bigram\n(1,2)','Uni+Bi+Tri\n(1,3)']
    algo_list = ["SVM", "Random Forest", "XGBoost"]
    colors = ['#4285F4', '#34A853', '#FBBC04']

    x     = np.arange(5)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, algo in enumerate(algo_list):
        if algo not in filter_algo:
            continue
        subset = df_hasil[df_hasil['Algoritma'] == algo]
        vals = []
        for ng in ngram_order:
            row = subset[subset['N-Gram'] == ng]
            vals.append(row['F1-Score'].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i*width, vals, width, label=algo,
                      color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.004,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Konfigurasi N-Gram', fontsize=11)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=11)
    ax.set_title('Perbandingan F1-Score: N-Gram vs Algoritma', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(ngram_labels, fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### 🏆 Model Terbaik Keseluruhan")
    best = df_hasil.loc[df_hasil['F1-Score'].idxmax()]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Algoritma", best['Algoritma'])
    col2.metric("N-Gram", "Uni+Bigram (1,2)")
    col3.metric("Accuracy", f"{best['Accuracy']:.2%}")
    col4.metric("F1-Score", f"{best['F1-Score']:.2%}")
    col5.metric("Waktu", f"{best['Waktu (s)']} detik")
