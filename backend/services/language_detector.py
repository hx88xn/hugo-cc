"""
Deterministic anchor-word language detector for the 5 supported call-center
languages: English (en), Urdu (ur), Punjabi (pa), Pashto (ps), Sindhi (sd).

Runs on Gemini Live input_audio_transcription text. Used as a second,
authoritative signal that can silently correct Gemini's own language pick
via the setResponseLanguage tool dispatcher.

Pure-Python, ~1ms on a 50-word turn. No external dependencies.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Per-language anchor word lexicons. Tokens are lowercased and stripped of
# punctuation before matching. Cover Roman transliteration AND native script
# (Urdu script tokens included for ur/pa/ps/sd since Gemini may transcribe
# either way).
LEXICON: dict[str, set[str]] = {
    "en": {
        # Function words and pronouns
        "the", "is", "are", "was", "were", "am", "be", "been", "being",
        "a", "an", "of", "to", "in", "on", "at", "for", "with", "from",
        "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our",
        "he", "she", "it", "they", "them", "their", "this", "that", "these", "those",
        "what", "when", "where", "why", "how", "which", "who", "whom",
        # Common verbs / call-center words
        "want", "need", "have", "has", "had", "do", "does", "did", "can", "could",
        "would", "should", "will", "tell", "give", "get", "make", "take", "go",
        "please", "thank", "thanks", "hello", "hi", "yes", "no", "okay", "ok",
        "balance", "account", "card", "loan", "finance", "rate", "profit",
        "minimum", "maximum", "amount", "limit", "deposit", "branch",
    },
    "ur": {
        # Roman Urdu function words / verb forms
        "hai", "hain", "ho", "hoga", "hogi", "tha", "thi", "the",
        "kya", "kyun", "kyon", "kaise", "kaisay", "kaisa", "kitna", "kitni",
        "mujhe", "mujhko", "mera", "meri", "mere", "main", "hum", "hamara",
        "aap", "apni", "apna", "apne", "tum", "tumhara",
        "ko", "ka", "ke", "se", "par", "mein",
        "shukriya", "meherbani", "baraye", "theek", "acha", "achha", "bilkul",
        "kar", "karna", "karta", "karti", "karte", "karen", "karein",
        "sakta", "sakti", "sakte", "raha", "rahi", "rahe", "raho",
        "samajh", "samjha", "samjhi", "samajhna",
        "batao", "bataye", "batayein", "bataiye",
        "chahiye", "chahta", "chahti",
        "nahin", "nahi", "haan", "ji",
        "kahan", "yahan", "wahan", "abhi", "phir",
        # Urdu-script common tokens
        "ہے", "ہیں", "کیا", "میں", "آپ", "مجھے", "کیسے", "کتنا",
        "شکریہ", "مہربانی", "ٹھیک", "اچھا", "نہیں", "ہاں", "کر", "کرنا",
        "کرتا", "کرتی", "سکتا", "سکتی", "بتائیں", "چاہیے",
    },
    "pa": {
        # Distinctively Punjabi (Roman)
        "tusi", "tussi", "tussin", "menu", "mainu", "mainoo",
        "tuhada", "tuhade", "tuhadi", "tuhadiyan",
        "kithay", "kithon", "kinna", "kinni", "kine",
        "haiga", "hega", "hegi", "haigi", "haige",
        "asi", "assi", "asaan", "saade", "saadi",
        "ainvein", "ainj", "ainjh", "edaan", "udaan",
        "hunda", "hundi", "hunde", "honda", "hondi",
        "changa", "changi", "change",
        "dasso", "dasna", "dasda", "dasdi", "dassiye",
        "vekho", "vekhna", "vekhda", "vekhdi",
        "kihda", "kihde",  # "ki" alone is too ambiguous (Urdu/Punjabi/Sindhi)
        "paisa", "paise",  # neutral, low weight
        "ohnu", "ohna", "ohde",
        "wich", "vich", "vichkar",
        "naal", "naa",
        "jaa", "jaake", "jaanda",
        "aaya", "aayi", "aaye",
        # Punjabi-script (Shahmukhi)
        "تُسی", "مینوں", "تُہاڈا", "ہیگا", "ہیگی", "اسی", "ساڈا",
        "ہندا", "ہندی", "چنگا", "دسو", "ویکھو",
    },
    "ps": {
        # Pashto (Roman)
        "tsa", "tsanga", "tsenga", "tso", "tsomra",
        "yam", "ye", "dai", "da", "di", "dey",
        "zama", "sta", "zmuzh", "zmuz", "stase", "staso",
        "khabara", "khabaray", "khpal", "khpala", "khpale",
        "khkuli", "khkula",
        "manana", "mannana",
        "kawom", "kawi", "kawu", "kram", "kre",
        "raza", "razi",
        "wakht", "wakhti",
        "cha",  # "che" removed — collides with Italian/French/etc.
        "pa", "ba",  # short — low weight on their own
        "ghwarham", "ghwaarham", "ghwaram",
        "owayem", "owayey", "wayey",
        "akhistal", "warkawom",
        "mehrabani", "mehrbani",
        # Pashto-script
        "څه", "زما", "ستا", "زمونږ", "ستاسو", "خبره", "خپل",
        "ښکلی", "مننه", "کوم", "کوي", "راځه", "وخت",
    },
    "sd": {
        # Sindhi (Roman)
        "chha", "chho", "chhi", "chhe",
        "aahyan", "aahyo", "aahe", "aahin", "aahyaan",
        "muhinjo", "muhinji", "muhinje",
        "tuhinjo", "tuhinji", "tuhinje",
        "asaanjo", "asaanji", "asaanje",
        "ketro", "ketri", "ketra",
        "thiyo", "thi", "thiyaan", "thiya",
        "vanjan", "vanjo", "vanjandus",
        "chayo", "chai", "chaye", "chavan",
        "hee", "huu", "hin", "hun",
        "achi", "acho",  # come (Sindhi) — distinct from Urdu acha
        "kahan", "kithe",  # ambiguous w/ pa, lower weight
        "saan", "sain",
        "wari", "wara",
        "budhi", "budho",  # heard
        # Sindhi-script
        "ڇا", "آھيان", "آھيو", "آھي", "مهنجو", "تهنجو", "اسانجو",
        "ڪيترو", "ٿيو", "ٿي", "وڃان", "چيو", "هي", "هو",
    },
}

# Per-language DISTINCTIVE markers — tokens that essentially never appear in
# the other 4 languages and so should count as strong evidence (weight 3
# instead of weight 1). Any one of these is usually enough to decide.
DISTINCTIVE: dict[str, set[str]] = {
    "en": set(),  # English is identified by mass of common words, not single markers
    "ur": {"kya", "kyun", "mujhe", "shukriya", "meherbani", "kaise", "kaisay",
           "chahiye", "samajh", "ٹھیک", "شکریہ", "مہربانی"},
    "pa": {"tusi", "tussi", "menu", "mainu", "tuhada", "tuhade",
           "haiga", "hega", "hegi", "haigi",
           "ainvein", "ainj", "dasso", "saade", "saadi", "hunda", "hundi",
           "تُسی", "مینوں", "تُہاڈا", "ہیگا", "ساڈا"},
    "ps": {"tsa", "zama", "sta", "zmuzh", "zmuz", "stase", "staso",
           "khpal", "khpala", "manana", "kawom", "ghwarham", "ghwaarham",
           "زما", "ستا", "زمونږ", "خپل", "مننه"},
    "sd": {"chha", "aahyan", "aahyo", "aahe", "muhinjo", "tuhinjo", "asaanjo",
           "thiyo", "vanjan", "ڇا", "آھيان", "آھيو", "آھي", "مهنجو", "تهنجو"},
}


# Tokens that appear across multiple of the South Asian languages and are not
# useful as evidence on their own. Excluded from scoring.
SHARED_AMBIGUOUS: set[str] = {
    "ji", "haan", "nahi", "nahin",
    "balance", "account", "card", "loan",
    "cnic", "atm", "pin", "tpin", "otp",
    "bankislami", "bank", "islami",
    "muaziz", "saarif", "saheb", "sahib",
    "allah", "inshallah", "mashallah", "assalamualaikum", "salaam",
    "ok", "okay", "hmm", "uh", "um", "uhh", "umm",
    "ek", "do", "teen", "char", "panj", "panch",  # numerals shared
    "rupees", "rupay", "rupaye",
}


_TOKEN_RE = re.compile(r"[^\s؀-ۿݐ-ݿऀ-ॿa-zA-Z']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Preserves Arabic/Urdu/Sindhi-script characters and Latin letters.
    """
    if not text:
        return []
    # NFC normalise so diacritics are consistent.
    text = unicodedata.normalize("NFC", text)
    # Replace anything that's not a word character (Latin or supported script) with space.
    cleaned = _TOKEN_RE.sub(" ", text)
    return [t.lower() for t in cleaned.split() if t]


def detect_language(text: str) -> tuple[Optional[str], float, dict[str, int]]:
    """Classify the input text into one of {en, ur, pa, ps, sd} or None.

    Returns (lang, confidence, raw_scores).
        - lang: best language ISO code, or None if no anchor word matched.
        - confidence: best_score / total_score, in [0, 1]. Higher = more decisive.
        - raw_scores: per-language hit counts, for logging.
    """
    tokens = _tokenize(text)
    scores: dict[str, int] = {lang: 0 for lang in LEXICON}
    if not tokens:
        return None, 0.0, scores

    for tok in tokens:
        if tok in SHARED_AMBIGUOUS:
            continue
        for lang, words in LEXICON.items():
            if tok in words:
                # Weight 3 if the token is distinctive for this language,
                # otherwise weight 1.
                scores[lang] += 3 if tok in DISTINCTIVE.get(lang, set()) else 1

    total = sum(scores.values())
    if total == 0:
        return None, 0.0, scores

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]
    # Require ≥ 2 score points (1 distinctive marker OR 2 generic hits) to
    # avoid a single coincidental match (e.g. Italian "che" leaking as Pashto).
    # Exception: a single-token transcript may decide on 1 generic hit
    # (e.g. "Tso?", "Chha?").
    non_shared_tokens = [t for t in tokens if t not in SHARED_AMBIGUOUS]
    if best_score < 2 and len(non_shared_tokens) > 1:
        return None, 0.0, scores

    confidence = best_score / total
    return best_lang, confidence, scores
