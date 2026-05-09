# preprocessing_utils.py
# AUTO-GENERATED dari Notebook_0_Preprocessing.ipynb
# Import file ini di app.py Streamlit — JANGAN edit manual

import re
import nltk

# Download NLTK data sebelum dipakai (aman dijalankan berulang kali)
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

NEGASI = {
    "tidak","tak","bukan","bukanlah","tidaklah","belum","belumlah",
    "tanpa","jangan","janganlah","jangankan",
    "enggak","nggak","ngga","gak","gk","ngk","gx","tdk",
    "nda","ndak","kagak","kaga","kagaklah",
    "belom","belon","blm","blum","tdak","tida","takkan",
    "ogah","gamau","gamao","gaingin","gkma","males","enggan",
    "kurang","jarang","sulit","susah","mustahil","minim","lemah",
    "gaaa","gakkk","nggakk","nggaaa","tdkk","tdakkk",
    "tidakkan","gakakan","gaakan","enggakakan",
}

NORMALISASI_NEGASI = {
    "gak":"tidak","ga":"tidak","gk":"tidak","ngk":"tidak",
    "gx":"tidak","tdk":"tidak","tdak":"tidak","tida":"tidak",
    "nda":"tidak","ndak":"tidak","tdkk":"tidak","tdakkk":"tidak",
    "gaaa":"tidak","gakkk":"tidak",
    "enggak":"tidak","nggak":"tidak","ngga":"tidak",
    "nggakk":"tidak","nggaaa":"tidak",
    "kagak":"tidak","kaga":"tidak","kagaklah":"tidak",
    "belom":"belum","belon":"belum","blm":"belum","blum":"belum",
    "gamau":"tidak_mau","gamao":"tidak_mau",
    "gaingin":"tidak_ingin","gkma":"tidak_mau",
    "takkan":"tidak_akan","tidakkan":"tidak_akan",
    "gakakan":"tidak_akan","gaakan":"tidak_akan","enggakakan":"tidak_akan",
    "tidaklah":"tidak","bukanlah":"bukan","belumlah":"belum",
    "janganlah":"jangan",
}

_factory  = StemmerFactory()
_stemmer  = _factory.create_stemmer()
_sw_base  = set(stopwords.words("indonesian"))
_custom   = {
    "yg","ny","nya","deh","sih","nih","aja","ajah",
    "udah","udh","dah","bgt","bngtt","bngt","dong","loh","lho","lah",
    "kalo","kalau","kalok","kali","kan","kok","yaa","yaaa","yap","yep",
    "webtoon","app","aplikasi","line","play","store","google","naver",
    "sekarang","skrg","skrang","tadi","barusan","besok","kemarin",
    "aku","saya","gue","gw","ane","kita","kami","kalian","mereka",
    "dia","kamu","lo","lu","mu",
}
STOP_FINAL = (_sw_base | _custom) - NEGASI


def preprocess_text(text):
    """Fungsi utama preprocessing — output string siap masuk TF-IDF/model."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [NORMALISASI_NEGASI.get(t, t) for t in tokens]
    tokens = [t for t in tokens if (t not in STOP_FINAL) and (len(t) > 1 or t in NEGASI)]
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGASI and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok not in NEGASI:
                ns = _stemmer.stem(next_tok) if "_" not in next_tok else next_tok
                result.append(f"{tok}_{ns}")
                result.append(tok)
                i += 2
                continue
        result.append(_stemmer.stem(tok) if "_" not in tok else tok)
        i += 1
    return " ".join(result)


# Alias agar kompatibel dengan kode lama yang pakai nama preprocess()
preprocess = preprocess_text


def preprocess_with_steps(text):
    """
    Preprocessing sama persis dengan preprocess_text(),
    tapi mengembalikan (final_text, steps_dict) untuk ditampilkan
    di UI Streamlit tahap per tahap.
    """
    steps = {}
    if not isinstance(text, str) or not text.strip():
        return "", {"original": text}

    steps['original'] = text

    # 1. Case folding
    t = text.lower().strip()
    steps['case_folding'] = t

    # 2. Normalisasi karakter berulang
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    steps['normalisasi'] = t

    # 3. Cleaning
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = t.encode("ascii", "ignore").decode()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    steps['cleaning'] = t

    # 4. Tokenisasi awal
    tokens = word_tokenize(t)
    steps['tokens_raw'] = tokens

    # 5. Normalisasi negasi slang (gak→tidak, dll)
    tokens = [NORMALISASI_NEGASI.get(tok, tok) for tok in tokens]
    steps['slang'] = ' '.join(tokens)

    # 6. Stopword removal (negasi dipertahankan)
    tokens = [tok for tok in tokens if (tok not in STOP_FINAL) and (len(tok) > 1 or tok in NEGASI)]
    steps['after_stopword'] = tokens

    # 6.5 Bigram negasi: tidak + kata → tidak_kata (sekaligus pertahankan tidak)
    result = []
    i = 0
    negasi_bigrams = []
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGASI and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok not in NEGASI:
                ns = _stemmer.stem(next_tok) if "_" not in next_tok else next_tok
                bigram = f"{tok}_{ns}"
                result.append(bigram)
                result.append(tok)
                negasi_bigrams.append(bigram)
                i += 2
                continue
        result.append(tok)
        i += 1
    steps['after_negasi_bigram'] = result
    steps['negasi_bigrams_dibuat'] = negasi_bigrams  # info tambahan

    # 7. Stemming (skip token yang sudah berupa bigram negasi)
    stemmed = [_stemmer.stem(tok) if "_" not in tok else tok for tok in result]
    steps['after_stemming'] = stemmed

    final = " ".join(stemmed)
    return final, steps
