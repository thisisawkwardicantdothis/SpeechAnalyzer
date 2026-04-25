def test_sentences_returns_result(simple_doc):
    from analyzers.sentences import SentencesAnalyzer
    result = SentencesAnalyzer().run(simple_doc)
    assert result.name == "sentences"
    assert "sentence_count" in result.metrics
    assert "avg_sentence_length_words" in result.metrics
    assert "median_sentence_length" in result.metrics
    assert "max_sentence_length" in result.metrics
    assert "sentence_length_std" in result.metrics


def test_sentences_count_positive(simple_doc):
    from analyzers.sentences import SentencesAnalyzer
    result = SentencesAnalyzer().run(simple_doc)
    assert result.metrics["sentence_count"] >= 1


def test_sentences_avg_positive(simple_doc):
    from analyzers.sentences import SentencesAnalyzer
    result = SentencesAnalyzer().run(simple_doc)
    assert result.metrics["avg_sentence_length_words"] > 0


def test_sentences_requires_pos():
    from analyzers.sentences import SentencesAnalyzer
    assert SentencesAnalyzer().requires_pos is True


def test_sentences_produces_figure(simple_doc):
    from analyzers.sentences import SentencesAnalyzer
    result = SentencesAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1
