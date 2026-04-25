def test_preprocess_creates_transcript_doc(simple_segments, monkeypatch):
    import spacy
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))
    from preprocessor import preprocess

    doc = preprocess(simple_segments, "en")

    assert doc.language == "en"
    assert doc.spacy_doc is not None
    assert len(list(doc.spacy_doc)) > 0
    assert "nlp" in doc.annotations


def test_preprocess_removes_duplicate_segments(monkeypatch):
    import spacy
    from analyzers.base import Segment
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))
    from preprocessor import preprocess

    segments = [
        Segment(0.0, 2.0, "hello world", 0.9),
        Segment(2.0, 4.0, "hello world", 0.8),
        Segment(4.0, 6.0, "unique text", 0.9),
    ]
    doc = preprocess(segments, "en")
    assert len(doc.segments) == 2
    assert doc.segments[0].text == "hello world"
    assert doc.segments[1].text == "unique text"


def test_preprocess_builds_raw_from_all_segments(monkeypatch):
    import spacy
    from analyzers.base import Segment
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))
    from preprocessor import preprocess

    segments = [
        Segment(0.0, 2.0, "hello", 0.9),
        Segment(2.0, 4.0, "hello", 0.8),
    ]
    doc = preprocess(segments, "en")
    assert "hello hello" == doc.raw_text
    assert "hello" == doc.clean_text


def test_preprocess_fallback_language_uses_xx_model(monkeypatch):
    import spacy
    loaded = {}

    def mock_load(name):
        loaded["model"] = name
        return spacy.load("en_core_web_sm")

    monkeypatch.setattr("preprocessor._load_spacy_model", mock_load)
    from preprocessor import preprocess
    from analyzers.base import Segment

    segments = [Segment(0.0, 1.0, "text", 1.0)]
    preprocess(segments, "zh")
    assert loaded["model"] == "zh"
