def test_pipeline_runs_selected_modules(simple_segments, monkeypatch):
    import spacy
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))

    from pipeline import run_pipeline
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            segments=simple_segments,
            language="en",
            modules=["speech_rate", "word_length"],
            output_dir=tmpdir,
        )

    names = [r.name for r in results]
    assert "speech_rate" in names
    assert "word_length" in names
    assert "vocabulary" not in names


def test_pipeline_skips_unsupported_language_pos_modules(monkeypatch):
    import spacy
    from analyzers.base import Segment
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))

    from pipeline import run_pipeline
    import tempfile

    segments = [Segment(0.0, 5.0, "some text here", 0.9)]
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            segments=segments,
            language="zh",
            modules=["vocabulary", "speech_rate"],
            output_dir=tmpdir,
        )

    names = [r.name for r in results]
    assert "speech_rate" in names
    vocab_result = next(r for r in results if r.name == "vocabulary")
    assert vocab_result.warnings


def test_pipeline_all_modules_run_for_supported_language(simple_segments, monkeypatch):
    import spacy
    monkeypatch.setattr("preprocessor._load_spacy_model", lambda lang: spacy.load("en_core_web_sm"))

    from pipeline import run_pipeline
    from analyzers import ALL_MODULES
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            segments=simple_segments,
            language="en",
            modules=ALL_MODULES,
            output_dir=tmpdir,
        )

    assert len(results) == len(ALL_MODULES)
