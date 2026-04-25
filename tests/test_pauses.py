def test_pauses_returns_result(simple_doc):
    from analyzers.pauses import PausesAnalyzer
    result = PausesAnalyzer().run(simple_doc)
    assert result.name == "pauses"
    assert "pause_count" in result.metrics
    assert "total_pause_seconds" in result.metrics
    assert "mean_pause_duration" in result.metrics
    assert "max_pause_duration" in result.metrics
    assert "silence_ratio" in result.metrics


def test_pauses_detects_gap(simple_doc):
    # conftest simple_segments has gap of 1.0s between segment 2 and 3
    from analyzers.pauses import PausesAnalyzer
    result = PausesAnalyzer().run(simple_doc)
    assert result.metrics["pause_count"] >= 1
    assert result.metrics["total_pause_seconds"] > 0


def test_pauses_silence_ratio_bounded(simple_doc):
    from analyzers.pauses import PausesAnalyzer
    result = PausesAnalyzer().run(simple_doc)
    assert 0.0 <= result.metrics["silence_ratio"] <= 1.0


def test_pauses_no_false_positives(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.pauses import PausesAnalyzer

    # Perfectly contiguous segments → no pauses
    segments = [
        Segment(0.0, 5.0, "hello world", 1.0),
        Segment(5.0, 10.0, "how are you", 1.0),
    ]
    text = "hello world how are you"
    doc = TranscriptDoc(text, text, segments, nlp(text), "en")
    result = PausesAnalyzer().run(doc)
    assert result.metrics["pause_count"] == 0


def test_pauses_does_not_require_pos():
    from analyzers.pauses import PausesAnalyzer
    assert PausesAnalyzer().requires_pos is False


def test_pauses_produces_figure(simple_doc):
    from analyzers.pauses import PausesAnalyzer
    result = PausesAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1
