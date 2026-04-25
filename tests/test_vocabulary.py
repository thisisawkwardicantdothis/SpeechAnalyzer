def test_vocabulary_returns_result(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert result.name == "vocabulary"
    assert "mattr" in result.metrics
    assert "ttr" in result.metrics
    assert "chao1" in result.metrics


def test_ttr_between_zero_and_one(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert 0.0 < result.metrics["ttr"] <= 1.0


def test_mattr_between_zero_and_one(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert 0.0 < result.metrics["mattr"] <= 1.0


def test_chao1_at_least_observed_types(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert result.metrics["chao1"] >= result.metrics["observed_types"]


def test_chao1_smoothing_warning_when_f2_zero(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.vocabulary import VocabularyAnalyzer

    # All unique words → f2=0
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    spacy_doc = nlp(text)
    segments = [Segment(0.0, 5.0, text, 1.0)]
    doc = TranscriptDoc(text, text, segments, spacy_doc, "en")

    result = VocabularyAnalyzer().run(doc)
    assert any("Smoothing" in w or "smoothing" in w.lower() for w in result.warnings)


def test_vocabulary_requires_pos():
    from analyzers.vocabulary import VocabularyAnalyzer
    assert VocabularyAnalyzer().requires_pos is True


def test_vocabulary_produces_figure(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert len(result.figures) >= 3


def test_vocabulary_new_metrics(simple_doc):
    from analyzers.vocabulary import VocabularyAnalyzer
    result = VocabularyAnalyzer().run(simple_doc)
    assert "hapax_percent" in result.metrics
    assert "top_10_words" in result.metrics
    assert 0.0 <= result.metrics["hapax_percent"] <= 100.0
    assert isinstance(result.metrics["top_10_words"], dict)
