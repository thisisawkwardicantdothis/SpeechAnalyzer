import spacy
from spacy.cli import download as spacy_download

from analyzers.base import Segment, TranscriptDoc, SUPPORTED_LANGUAGES

LANGUAGE_MODELS = {
    "de": "de_core_news_lg",
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "it": "it_core_news_lg",
    "nl": "nl_core_news_lg",
    "pt": "pt_core_news_lg",
}
FALLBACK_MODEL = "xx_ent_wiki_sm"


def _load_spacy_model(language: str) -> spacy.language.Language:
    model_name = LANGUAGE_MODELS.get(language, FALLBACK_MODEL)
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)


def _clean_segments(segments: list) -> list:
    seen: set = set()
    cleaned = []
    for seg in segments:
        text = seg.text.strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(Segment(seg.start, seg.end, text, seg.confidence))
    return cleaned


def preprocess(segments: list, language: str) -> TranscriptDoc:
    cleaned = _clean_segments(segments)
    raw_text = " ".join(seg.text for seg in segments)
    clean_text = " ".join(seg.text for seg in cleaned)

    nlp = _load_spacy_model(language)
    spacy_doc = nlp(clean_text)

    return TranscriptDoc(
        raw_text=raw_text,
        clean_text=clean_text,
        segments=cleaned,
        spacy_doc=spacy_doc,
        language=language,
        annotations={"nlp": nlp},
    )
