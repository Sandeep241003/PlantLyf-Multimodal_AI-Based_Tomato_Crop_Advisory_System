from langdetect import detect
from deep_translator import GoogleTranslator

INDIAN_LANGUAGES = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam"
}

def detect_language(text):
    try:
        lang = detect(text)
        if lang in INDIAN_LANGUAGES:
            return lang
        else:
            return "en"
    except:
        return "en"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text