
# preprocessing_utils.py
# AUTO-GENERATED dari Notebook_0_Preprocessing.ipynb
# Import file ini di app.py Streamlit — JANGAN edit manual

import re
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

def preprocess(text):
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
