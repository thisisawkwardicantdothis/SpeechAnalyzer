import pytest

def test_speech_rate_returns_result(simple_doc):
    from analyzers.speech_rate import SpeechRateAnalyzer
    result = SpeechRateAnalyzer().run(simple_doc)
    assert result.name == "speech_rate"
    assert "wpm_net" in result.metrics
    assert "wpm_gross" in result.metrics
    assert "total_tokens" in result.metrics
    assert "net_speech_seconds" in result.metrics
    assert "gross_duration_seconds" in result.metrics


def test_net_wpm_positive(simple_doc):
    from analyzers.speech_rate import SpeechRateAnalyzer
    result = SpeechRateAnalyzer().run(simple_doc)
    assert result.metrics["wpm_net"] > 0


def test_net_wpm_excludes_low_confidence(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.speech_rate import SpeechRateAnalyzer

    # Segment with low confidence should be excluded from net speech time
    segments = [
        Segment(0.0, 60.0, "good segment " * 10, confidence=0.9),
        Segment(60.0, 120.0, "bad segment " * 10, confidence=0.3),
    ]
    text = " ".join(s.text for s in segments)
    doc = TranscriptDoc(text, text, segments, nlp(text), "en")

    result = SpeechRateAnalyzer().run(doc)
    # net only uses the 60s high-confidence segment
    assert result.metrics["net_speech_seconds"] == pytest.approx(60.0, abs=1.0)


def test_gross_wpm_uses_full_duration(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.speech_rate import SpeechRateAnalyzer

    segments = [Segment(0.0, 120.0, "word " * 120, confidence=0.9)]
    text = "word " * 120
    doc = TranscriptDoc(text, text, segments, nlp(text), "en")

    result = SpeechRateAnalyzer().run(doc)
    assert result.metrics["gross_duration_seconds"] == pytest.approx(120.0, abs=1.0)


def test_speech_rate_does_not_require_pos():
    from analyzers.speech_rate import SpeechRateAnalyzer
    assert SpeechRateAnalyzer().requires_pos is False


def test_speech_rate_produces_figure(simple_doc):
    from analyzers.speech_rate import SpeechRateAnalyzer
    result = SpeechRateAnalyzer().run(simple_doc)
    assert len(result.figures) >= 3


def test_speech_rate_new_metrics(simple_doc):
    from analyzers.speech_rate import SpeechRateAnalyzer
    result = SpeechRateAnalyzer().run(simple_doc)
    assert "wpm_std" in result.metrics
    assert "pause_count" in result.metrics
    assert "total_pause_seconds" in result.metrics
    assert "silence_ratio" in result.metrics
    assert result.metrics["pause_count"] >= 0
    assert 0.0 <= result.metrics["silence_ratio"] <= 1.0
