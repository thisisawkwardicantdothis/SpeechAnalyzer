def test_word_length_returns_result(simple_doc):
    from analyzers.word_length import WordLengthAnalyzer
    result = WordLengthAnalyzer().run(simple_doc)
    assert result.name == "word_length"
    assert "mean_word_length" in result.metrics
    assert "median_word_length" in result.metrics
    assert "max_word_length" in result.metrics


def test_mean_word_length_positive(simple_doc):
    from analyzers.word_length import WordLengthAnalyzer
    result = WordLengthAnalyzer().run(simple_doc)
    assert result.metrics["mean_word_length"] > 0


def test_word_length_does_not_require_pos():
    from analyzers.word_length import WordLengthAnalyzer
    assert WordLengthAnalyzer().requires_pos is False


def test_word_length_excludes_stopwords(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.word_length import WordLengthAnalyzer

    text = "I enjoy a programming"
    spacy_doc = nlp(text)
    doc = TranscriptDoc(text, text, [Segment(0.0, 2.0, text, 1.0)], spacy_doc, "en")
    result = WordLengthAnalyzer().run(doc)
    assert result.metrics["mean_word_length"] > 5.0


def test_word_length_produces_figure(simple_doc):
    from analyzers.word_length import WordLengthAnalyzer
    result = WordLengthAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1
