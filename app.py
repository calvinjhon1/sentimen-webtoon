import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
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
        with open('model_SVM_best.pkl', 'rb') as f:
            saved = pickle.load(f)
        return saved['model'], saved['tfidf']
    except FileNotFoundError:
        return None, None

@st.cache_resource
def load_preprocessing():
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt',     quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        sw = set(stopwords.words('indonesian'))

        # Kata negasi WAJIB dipertahankan
        negasi = {
            'tidak','tak','bukan','bukanlah','tidaklah',
            'kurang','jangan','jangankan','janganlah',
            'belum','belumlah','tanpa','enggak','nggak','ngga'
        }
        sw = sw - negasi

        custom_sw = {
            'yg','ny','nya','deh','sih','nih','aja','ajah',
            'udah','udh','dah','bgt','bngtt','bngt','dong',
            'loh','lho','lah','yaa','yaaa','yap','yep',
            'webtoon','app','aplikasi','line','play',
            'store','google','naver','skrg','skrang',
            'barusan','gue','gw','ane','lo','lu',
        }
        sw = sw.union(custom_sw)

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer, sw, word_tokenize
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return None, None, None

# Kamus normalisasi slang
SLANG = {
    'ga':'tidak','gak':'tidak','gk':'tidak',
    'ngga':'tidak','nggak':'tidak','tdk':'tidak',
    'gabisa':'tidak bisa','gaada':'tidak ada',
    'gabuka':'tidak buka','gajalan':'tidak jalan',
    'bgt':'banget','bngt':'banget','bngtt':'banget',
    'kalo':'kalau','klo':'kalau',
    'krn':'karena','krna':'karena',
    'yg':'yang','udh':'sudah','udah':'sudah',
    'blm':'belum','blum':'belum',
    'tp':'tapi','tpi':'tapi',
    'gmn':'bagaimana','knp':'kenapa',
    'emg':'memang','emang':'memang',
    'aja':'saja','doang':'saja','doank':'saja',
    'bs':'bisa','bsa':'bisa',
    'lg':'lagi','lgi':'lagi',
    'sdh':'sudah','dah':'sudah',
    'hrs':'harus','hrs':'harus',
    'msh':'masih','masi':'masih',
    'jgn':'jangan','jd':'jadi',
    'sm':'sama','br':'baru',
    'gt':'gitu','gitu':'begitu',
    'nih':'ini','tuh':'itu',
    'pdhl':'padahal','krna':'karena',
    'sampe':'sampai','smpe':'sampai',
    'banget':'banget','bener':'benar',
    'susah':'susah','capek':'capek',
}

def preprocess_with_steps(text, stemmer, stop_words, tokenize):
    steps = {}
    steps['original'] = str(text)

    # 1. Case folding
    t = str(text).lower()
    steps['case_folding'] = t

    # 2. Normalisasi karakter berulang
    t = re.sub(r'(.)\1{2,}', r'\1', t)
    steps['normalisasi'] = t

    # 3. Hapus URL, non-ASCII, angka, tanda baca
    t = re.sub(r'http\S+|www\S+', '', t)
    t = t.encode('ascii', 'ignore').decode('ascii')
    t = re.sub(r'\d+', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    steps['cleaning'] = t

    # 4. Normalisasi slang
    words = t.split()
    t = ' '.join([SLANG.get(w, w) for w in words])
    steps['slang'] = t

    # 5. Tokenisasi
    tokens = tokenize(t)
    steps['tokens_raw'] = tokens

    # 6. Stopword removal
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]
    steps['after_stopword'] = tokens

    # 7. Stemming
    tokens = [stemmer.stem(tok) for tok in tokens]
    steps['after_stemming'] = tokens

    final = ' '.join(tokens)
    return final, steps

# ── Data hasil penelitian ─────────────────────────────────────────────────────
HASIL = [
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Webtoon_App_Logo.png/240px-Webtoon_App_Logo.png", width=80)
    st.markdown("### 📚 Analisis Sentimen Webtoon")
    st.markdown("---")
    menu = st.radio("Navigasi", ["🏠 Beranda", "🔍 Prediksi Sentimen", "📊 Hasil Penelitian"])
    st.markdown("---")
    st.markdown("**Model Terbaik:**")
    st.markdown("✅ SVM + Unigram+Bigram")
    st.markdown("**Akurasi:** 91.53%")
    st.markdown("**F1-Score:** 91.32%")
    st.markdown("---")
    st.markdown("**Peneliti:** Calvin Jhon")
    st.markdown("Sistem Informasi · 2025")

# ══════════════════════════════════════════════════════════════════════════════
# BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown("## 📚 Sistem Analisis Sentimen Ulasan Aplikasi Webtoon")
    st.markdown("*Implementasi Model SVM + TF-IDF Unigram+Bigram — Performa Terbaik dari 15 Eksperimen*")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        ["91.53%","91.24%","91.53%","91.32%"],
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
        st.success("**SVM + Unigram+Bigram (1,2)**")
        for s in ["1️⃣ Case Folding","2️⃣ Normalisasi karakter berulang",
                  "3️⃣ Cleaning (URL, emoji, angka)","4️⃣ Normalisasi slang (ga→tidak)",
                  "5️⃣ Tokenisasi","6️⃣ Stopword Removal (negasi dipertahankan)",
                  "7️⃣ Stemming — PySastrawi","8️⃣ TF-IDF + Unigram+Bigram","9️⃣ SVM Linear"]:
            st.markdown(f'<div class="step-box">{s}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Ringkasan Temuan")
    ca, cb, cc = st.columns(3)
    ca.success("**Temuan 1**\n\nUnigram+Bigram (1,2) terbaik pada SVM & RF — menangkap konteks frasa tanpa sparsity berlebihan.")
    cb.info("**Temuan 2**\n\nSVM unggul keseluruhan (F1 91,32%). XGBoost lebih seimbang mendeteksi kelas negatif berkat scale_pos_weight.")
    cc.warning("**Temuan 3**\n\nTrigram murni (3,3) konsisten paling rendah di semua algoritma akibat sparsity tinggi pada ulasan pendek.")

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
	@@ -263,21 +262,21 @@ def preprocess(text, stemmer, stop_words, tokenize):
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
	@@ -286,38 +285,38 @@ def preprocess(text, stemmer, stop_words, tokenize):
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
                    'Unigram+Bigram (1,2) ★','Unigram+Bigram+Trigram (1,3)']
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
    ax.set_ylim(0.6, 1.02)
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
    b1.metric("Algoritma",  best['Algoritma'])
    b2.metric("N-Gram",     "Uni+Bigram (1,2)")
    b3.metric("Accuracy",   f"{best['Accuracy']:.2%}")
    b4.metric("F1-Score",   f"{best['F1-Score']:.2%}")
    b5.metric("Waktu Training", f"{best['Waktu (s)']} detik")
