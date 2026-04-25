def test_wordcloud_returns_result(simple_doc):
    from analyzers.wordcloud_gen import WordcloudAnalyzer
    result = WordcloudAnalyzer().run(simple_doc)
    assert result.name == "wordcloud"
    assert "top_words" in result.metrics


def test_wordcloud_requires_pos():
    from analyzers.wordcloud_gen import WordcloudAnalyzer
    assert WordcloudAnalyzer().requires_pos is True


def test_wordcloud_produces_figure(simple_doc):
    from analyzers.wordcloud_gen import WordcloudAnalyzer
    result = WordcloudAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1


def test_wordcloud_top_words_non_empty(simple_doc):
    from analyzers.wordcloud_gen import WordcloudAnalyzer
    result = WordcloudAnalyzer().run(simple_doc)
    assert len(result.metrics["top_words"]) > 0


def test_wordcloud_fallback_single_segment(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.wordcloud_gen import WordcloudAnalyzer

    text = "Python programming language code"
    spacy_doc = nlp(text)
    doc = TranscriptDoc(text, text, [Segment(0.0, 3.0, text, 1.0)], spacy_doc, "en",
                        annotations={"nlp": nlp})
    result = WordcloudAnalyzer().run(doc)
    assert result.metrics["top_words"]
