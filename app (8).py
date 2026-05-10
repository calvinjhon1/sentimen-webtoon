import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analisis Sentimen Webtoon",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
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
    margin-bottom: 1rem;
}
.result-negative {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    color: white; padding: 1.5rem; border-radius: 10px;
    text-align: center; font-size: 1.5rem; font-weight: bold;
    margin-bottom: 1rem;
}
.info-box {
    background: #f0f2f6; padding: 1rem;
    border-radius: 8px; border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.step-box {
    background: white; padding: 0.7rem 1rem;
    border-radius: 8px; margin: 0.3rem 0;
    border: 1px solid #e0e0e0; font-size: 0.9rem;
}
.warning-box {
    background: #fff3cd; padding: 0.9rem 1rem;
    border-radius: 8px; border-left: 4px solid #ffc107;
    margin: 0.8rem 0; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('model_SVM_best (2).pkl', 'rb') as f:
            saved = pickle.load(f)
        return saved['model'], saved['tfidf']
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_preprocessing():
    """
    Import fungsi preprocessing dari preprocessing_utils.py.
    File ini di-generate otomatis dari notebook sehingga
    IDENTIK dengan preprocessing yang dipakai saat training.
    """
    # Download NLTK data dulu sebelum preprocessing_utils di-import,
    # karena preprocessing_utils memanggil stopwords.words() di level modul.
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)

    try:
        from preprocessing_utils import preprocess_text, preprocess_with_steps
        return preprocess_text, preprocess_with_steps
    except ImportError:
        st.error(
            "⚠️ File `preprocessing_utils.py` tidak ditemukan di folder yang sama dengan `app.py`. "
            "Pastikan sudah mengupload file tersebut dari Google Drive ke repo GitHub."
        )
        return None, None

# ── Data hasil penelitian ─────────────────────────────────────────────────────
HASIL = [
    {"Kode":"EXP-01","Algoritma":"SVM","N-Gram":"Unigram (1,1)","Accuracy":0.8309,"Precision":0.8300,"Recall":0.8309,"F1-Score":0.8304,"Waktu (s)":18.86},
    {"Kode":"EXP-02","Algoritma":"SVM","N-Gram":"Bigram (2,2)","Accuracy":0.7776,"Precision":0.7735,"Recall":0.7776,"F1-Score":0.7742,"Waktu (s)":14.38},
    {"Kode":"EXP-03","Algoritma":"SVM","N-Gram":"Trigram (3,3)","Accuracy":0.6995,"Precision":0.7019,"Recall":0.6995,"F1-Score":0.6482,"Waktu (s)":2.09},
    {"Kode":"EXP-04","Algoritma":"SVM","N-Gram":"Unigram+Bigram (1,2)","Accuracy":0.8286,"Precision":0.8284,"Recall":0.8286,"F1-Score":0.8285,"Waktu (s)":7.19},
    {"Kode":"EXP-05","Algoritma":"SVM","N-Gram":"Unigram+Bigram+Trigram (1,3) ★","Accuracy":0.8309,"Precision":0.8301,"Recall":0.8309,"F1-Score":0.8305,"Waktu (s)":5.95},
    {"Kode":"EXP-06","Algoritma":"Random Forest","N-Gram":"Unigram (1,1)","Accuracy":0.8217,"Precision":0.8193,"Recall":0.8217,"F1-Score":0.8197,"Waktu (s)":11.67},
    {"Kode":"EXP-07","Algoritma":"Random Forest","N-Gram":"Bigram (2,2)","Accuracy":0.7655,"Precision":0.7605,"Recall":0.7655,"F1-Score":0.7613,"Waktu (s)":15.92},
    {"Kode":"EXP-08","Algoritma":"Random Forest","N-Gram":"Trigram (3,3)","Accuracy":0.7006,"Precision":0.6922,"Recall":0.7006,"F1-Score":0.6624,"Waktu (s)":10.88},
    {"Kode":"EXP-09","Algoritma":"Random Forest","N-Gram":"Unigram+Bigram (1,2)","Accuracy":0.8217,"Precision":0.8198,"Recall":0.8217,"F1-Score":0.8203,"Waktu (s)":5.30},
    {"Kode":"EXP-10","Algoritma":"Random Forest","N-Gram":"Unigram+Bigram+Trigram (1,3) ★","Accuracy":0.8222,"Precision":0.8201,"Recall":0.8222,"F1-Score":0.8205,"Waktu (s)":5.63},
    {"Kode":"EXP-11","Algoritma":"XGBoost","N-Gram":"Unigram (1,1) ★","Accuracy":0.8089,"Precision":0.8178,"Recall":0.8089,"F1-Score":0.8114,"Waktu (s)":24.61},
    {"Kode":"EXP-12","Algoritma":"XGBoost","N-Gram":"Bigram (2,2)","Accuracy":0.6821,"Precision":0.7554,"Recall":0.6821,"F1-Score":0.6881,"Waktu (s)":7.31},
    {"Kode":"EXP-13","Algoritma":"XGBoost","N-Gram":"Trigram (3,3)","Accuracy":0.4592,"Precision":0.7157,"Recall":0.4592,"F1-Score":0.3981,"Waktu (s)":2.57},
    {"Kode":"EXP-14","Algoritma":"XGBoost","N-Gram":"Unigram+Bigram (1,2)","Accuracy":0.8037,"Precision":0.8148,"Recall":0.8037,"F1-Score":0.8066,"Waktu (s)":7.74},
    {"Kode":"EXP-15","Algoritma":"XGBoost","N-Gram":"Unigram+Bigram+Trigram (1,3)","Accuracy":0.8043,"Precision":0.8171,"Recall":0.8043,"F1-Score":0.8074,"Waktu (s)":10.20},
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Webtoon_App_Logo.png/240px-Webtoon_App_Logo.png", width=80)
    st.markdown("### 📚 Analisis Sentimen Webtoon")
    st.markdown("---")
    menu = st.radio("Navigasi", ["🏠 Beranda", "🔍 Prediksi Sentimen", "📊 Hasil Penelitian"])
    st.markdown("---")
    st.markdown("**Model Terbaik:**")
    st.markdown("✅ SVM + Uni+Bi+Trigram (1,3)")
    st.markdown("**Akurasi:** 83.09%")
    st.markdown("**F1-Score:** 83.05%")
    st.markdown("---")
    st.markdown("**Peneliti:** Calvin Jhon")
    st.markdown("Sistem Informasi · 2025")

# ══════════════════════════════════════════════════════════════════════════════
# BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown("## 📚 Sistem Analisis Sentimen Ulasan Aplikasi Webtoon")
    st.markdown("*Implementasi Model SVM + TF-IDF Unigram+Bigram+Trigram — Performa Terbaik dari 15 Eksperimen*")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        ["83.09%","83.01%","83.09%","83.05%"],
        ["Accuracy","Precision","Recall","F1-Score"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### 📋 Tentang Penelitian")
        st.markdown('<div class="info-box">Penelitian ini menganalisis pengaruh 5 konfigurasi n-gram pada TF-IDF terhadap performa 3 algoritma klasifikasi (SVM, Random Forest, XGBoost) dalam mengklasifikasikan sentimen ulasan aplikasi Webtoon di Google Play Store — menghasilkan 15 skenario eksperimen.</div>', unsafe_allow_html=True)
        st.markdown("**Rumusan Masalah:**")
        st.markdown("1. Bagaimana pengaruh konfigurasi n-gram terhadap performa TF-IDF?")
        st.markdown("2. Bagaimana perbandingan performa SVM, Random Forest, dan XGBoost?")
        st.markdown("3. Kombinasi mana yang menghasilkan performa terbaik?")
        c_a, c_b, c_c = st.columns(3)
        c_a.info("**Dataset**\n\n10.000 ulasan\nGoogle Play")
        c_b.info("**Algoritma**\n\nSVM · RF\nXGBoost")
        c_c.info("**N-Gram**\n\n5 konfigurasi\n15 eksperimen")

    with right:
        st.markdown("### ⚙️ Pipeline Model Terbaik")
        st.success("**SVM + Unigram+Bigram+Trigram (1,3)**")
        for s in [
            "1️⃣ Case Folding",
            "2️⃣ Normalisasi karakter berulang",
            "3️⃣ Cleaning (URL, emoji, angka)",
            "4️⃣ Normalisasi slang (ga→tidak)",
            "5️⃣ Tokenisasi",
            "6️⃣ Stopword Removal (negasi dipertahankan)",
            "6.5️⃣ Bigram Negasi (tidak_bagus, tidak_bisa, ...)",
            "7️⃣ Stemming — PySastrawi",
            "8️⃣ TF-IDF + Unigram+Bigram",
            "9️⃣ SVM Linear"
        ]:
            st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Ringkasan Temuan")
    ca, cb, cc = st.columns(3)
    ca.success("**Temuan 1**\n\nUnigram+Bigram+Trigram (1,3) terbaik pada SVM — kombinasi n-gram lengkap memberikan representasi fitur paling kaya.")
    cb.info("**Temuan 2**\n\nSVM unggul keseluruhan (F1 83.05%). Random Forest kompetitif di konfigurasi Unigram+Bigram+Trigram dengan F1 82.05%.")
    cc.warning("**Temuan 3**\n\nTrigram murni (3,3) konsisten paling rendah di semua algoritma — terutama XGBoost hanya F1 39.81% akibat sparsity tinggi.")

# ══════════════════════════════════════════════════════════════════════════════
# PREDIKSI SENTIMEN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 Prediksi Sentimen":
    st.markdown("## 🔍 Prediksi Sentimen Ulasan Webtoon")

    model, tfidf = load_model()
    preprocess_text, preprocess_with_steps = load_preprocessing()

    if model is None:
        st.error("⚠️ File `model_SVM_best.pkl` tidak ditemukan.")
        st.stop()
    if preprocess_with_steps is None:
        st.error("⚠️ `preprocessing_utils.py` gagal dimuat.")
        st.stop()

    # Catatan keterbatasan model
    # (dihapus — data sudah lebih seimbang dengan Most Relevant scraping)

    st.markdown("### ✍️ Masukkan Ulasan")

    contoh = {
        "Tulis sendiri...": "",
        "✅ Positif — pujian fitur":        "aplikasi ini bagus banget banyak komik seru yang bisa dibaca gratis",
        "✅ Positif — puas konten":          "webtoon sangat seru ceritanya menarik dan gambarnya keren sekali",
        "❌ Negatif — error teknis":         "aplikasi sering error dan lambat sekali sangat mengecewakan",
        "❌ Negatif — kecewa konten":        "kecewa sekali komiknya membosankan dan tidak menarik sama sekali",
        "❌ Negatif — tidak bisa login":     "gak bisa login sudah lama belum ada perbaikan dari developer",
        "❌ Negatif — iklan berlebihan":     "iklannya terlalu banyak dan sangat mengganggu tidak nyaman dipakai",
        "❌ Negatif — negasi (uji model)":  "aplikasi ini gak bagus sama sekali",
    }

    pilihan = st.selectbox("Atau pilih contoh:", list(contoh.keys()))
    user_input = st.text_area(
        "Teks ulasan:",
        value=contoh[pilihan],
        height=110,
        placeholder="Ketik ulasan Webtoon di sini..."
    )

    if st.button("🔮 Prediksi", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner("Memproses..."):
                # preprocess_with_steps dari preprocessing_utils mengembalikan (final_text, steps_dict)
                clean, steps = preprocess_with_steps(user_input)

            if not clean.strip():
                st.error("❌ Teks tidak mengandung kata bermakna setelah preprocessing.")
            else:
                vec  = tfidf.transform([clean])

                # Hitung probabilitas — cara berbeda tergantung tipe model
                if hasattr(model, 'predict_proba'):
                    # Random Forest & XGBoost → langsung punya predict_proba
                    proba    = model.predict_proba(vec)[0]
                    prob_pos = float(proba[1])
                    prob_neg = float(proba[0])
                else:
                    # SVM → pakai decision_function lalu konversi sigmoid
                    score    = model.decision_function(vec)[0]
                    prob_pos = 1 / (1 + math.exp(-score))
                    prob_neg = 1 - prob_pos

                pred_label = "POSITIF" if prob_pos >= 0.5 else "NEGATIF"

                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi")

                col_hasil, col_detail = st.columns([1, 1.3])

                with col_hasil:
                    if pred_label == "POSITIF":
                        st.markdown('<div class="result-positive">✅ POSITIF</div>', unsafe_allow_html=True)
                        st.success("Ulasan diklasifikasikan sebagai **sentimen positif**.")
                    else:
                        st.markdown('<div class="result-negative">❌ NEGATIF</div>', unsafe_allow_html=True)
                        st.error("Ulasan diklasifikasikan sebagai **sentimen negatif**.")

                    st.markdown("**Skor Kepercayaan Model:**")
                    st.metric("Skor → Positif", f"{prob_pos:.1%}")
                    st.metric("Skor → Negatif", f"{prob_neg:.1%}")

                    # Bar chart confidence
                    fig, ax = plt.subplots(figsize=(4, 1.8))
                    bars = ax.barh(
                        ['Negatif','Positif'],
                        [prob_neg, prob_pos],
                        color=['#f5576c','#38ef7d'],
                        edgecolor='white', linewidth=0.5
                    )
                    for bar, val in zip(bars, [prob_neg, prob_pos]):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                f'{val:.1%}', va='center', fontsize=10)
                    ax.set_xlim(0, 1.15)
                    ax.set_xlabel('Skor')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_detail:
                    st.markdown("**Detail Preprocessing Bertahap:**")
                    with st.expander("🔍 Lihat setiap tahapan preprocessing", expanded=True):

                        # Kunci yang diharapkan dari preprocessing_utils.preprocess_with_steps
                        # Sesuaikan label jika nama key berbeda di utils kamu
                        key_label_map = [
                            ('original',          'Teks asli'),
                            ('case_folding',       '1. Case folding'),
                            ('normalisasi',        '2. Normalisasi karakter berulang'),
                            ('cleaning',           '3. Cleaning'),
                            ('slang',              '4. Slang normalized'),
                            ('tokens_raw',         '5. Tokens'),
                            ('after_stopword',     '6. Stopword removal'),
                            ('after_negasi_bigram','6.5. Bigram negasi'),   # ← BARU
                            ('after_stemming',     '7. Stemming'),
                        ]

                        for key, label in key_label_map:
                            val = steps.get(key)
                            if val is None:
                                continue  # skip jika key tidak ada di utils
                            col_l, col_v = st.columns([1.4, 2])
                            col_l.markdown(f"**{label}:**")
                            col_v.markdown(str(val))

                        # Final baris terpisah
                        col_l, col_v = st.columns([1.4, 2])
                        col_l.markdown("**Final input model:**")
                        col_v.markdown(f"`{clean}`")

                        col_l, col_v = st.columns([1.4, 2])
                        col_l.markdown("**Jumlah token:**")
                        col_v.markdown(str(len(clean.split())))

                    st.info("**Model:** SVM + Unigram+Bigram+Trigram (1,3)\n\n**Accuracy:** 83.09% | **F1-Score:** 83.05%")

# ══════════════════════════════════════════════════════════════════════════════
# HASIL PENELITIAN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Hasil Penelitian":
    st.markdown("## 📊 Hasil Penelitian — 15 Eksperimen")
    st.markdown("Perbandingan performa seluruh kombinasi konfigurasi n-gram dan algoritma klasifikasi.")

    df = pd.DataFrame(HASIL)

    c1, c2 = st.columns(2)
    with c1:
        filter_algo = st.multiselect(
            "Filter Algoritma:",
            ["SVM","Random Forest","XGBoost"],
            default=["SVM","Random Forest","XGBoost"]
        )
    with c2:
        sort_by = st.selectbox("Urutkan berdasarkan:", ["F1-Score","Accuracy","Precision","Recall"])

    df_show = df[df['Algoritma'].isin(filter_algo)].sort_values(sort_by, ascending=False).reset_index(drop=True)
    df_show.index += 1
    st.dataframe(
        df_show[['Kode','Algoritma','N-Gram','Accuracy','Precision','Recall','F1-Score','Waktu (s)']],
        use_container_width=True, height=430
    )
    st.markdown("*★ = Konfigurasi terbaik per algoritma*")

    st.markdown("---")
    st.markdown("### 📈 Perbandingan F1-Score per Konfigurasi N-Gram")

    ngram_order  = ['Unigram (1,1)','Bigram (2,2)','Trigram (3,3)',
                    'Unigram+Bigram (1,2)','Unigram+Bigram+Trigram (1,3) ★']
    ngram_labels = ['Unigram\n(1,1)','Bigram\n(2,2)','Trigram\n(3,3)',
                    'Uni+Bigram\n(1,2)','Uni+Bi+Tri\n(1,3)']
    algo_list = ["SVM","Random Forest","XGBoost"]
    colors    = ['#4285F4','#34A853','#FBBC04']

    x     = np.arange(5)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, algo in enumerate(algo_list):
        if algo not in filter_algo:
            continue
        sub  = df[df['Algoritma'] == algo]
        vals = [sub[sub['N-Gram'] == ng]['F1-Score'].values[0]
                if len(sub[sub['N-Gram'] == ng]) > 0 else 0
                for ng in ngram_order]
        bars = ax.bar(x + i*width, vals, width, label=algo,
                      color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xlabel('Konfigurasi N-Gram', fontsize=11)
    ax.set_ylabel('F1-Score (Weighted)', fontsize=11)
    ax.set_title('Perbandingan F1-Score: Konfigurasi N-Gram vs Algoritma',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(ngram_labels, fontsize=10)
    ax.set_ylim(0.3, 0.95)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### 🏆 Model Terbaik Keseluruhan")
    best = df.loc[df['F1-Score'].idxmax()]
    b1,b2,b3,b4,b5 = st.columns(5)
    b1.metric("Algoritma",      best['Algoritma'])
    b2.metric("N-Gram",         "Uni+Bi+Trigram (1,3)")
    b3.metric("Accuracy",       f"{best['Accuracy']:.2%}")
    b4.metric("F1-Score",       f"{best['F1-Score']:.2%}")
    b5.metric("Waktu Training",  f"{best['Waktu (s)']} detik")
