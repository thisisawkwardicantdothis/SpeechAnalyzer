import json
from pathlib import Path


def test_exporter_creates_directory_structure(tmp_path, simple_doc):
    from exporter import export
    from analyzers.base import AnalyzerResult

    results = [
        AnalyzerResult(name="test", metrics={"x": 1.0}, figures=[], summary="OK", warnings=[])
    ]
    export(simple_doc, results, str(tmp_path / "out"))

    assert (tmp_path / "out" / "data").is_dir()
    assert (tmp_path / "out" / "reports").is_dir()
    assert (tmp_path / "out" / "visuals").is_dir()


def test_exporter_writes_transcripts(tmp_path, simple_doc):
    from exporter import export
    from analyzers.base import AnalyzerResult

    export(simple_doc, [], str(tmp_path / "out"))

    raw = (tmp_path / "out" / "data" / "transcript_raw.txt").read_text(encoding="utf-8")
    clean = (tmp_path / "out" / "data" / "transcript_clean.txt").read_text(encoding="utf-8")
    assert "0.0s" in raw
    assert simple_doc.segments[0].text in raw
    assert simple_doc.segments[0].text in clean


def test_exporter_writes_metrics_json(tmp_path, simple_doc):
    from exporter import export
    from analyzers.base import AnalyzerResult

    results = [
        AnalyzerResult(name="vocab", metrics={"ttr": 0.75}, figures=[], summary="ok")
    ]
    export(simple_doc, results, str(tmp_path / "out"))

    data = json.loads((tmp_path / "out" / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert data["vocab"]["ttr"] == 0.75


def test_exporter_writes_report(tmp_path, simple_doc):
    from exporter import export
    from analyzers.base import AnalyzerResult

    results = [
        AnalyzerResult(name="test", metrics={}, figures=[], summary="Test summary", warnings=["warn1"])
    ]
    export(simple_doc, results, str(tmp_path / "out"))

    report = (tmp_path / "out" / "reports" / "report.txt").read_text(encoding="utf-8")
    assert "Test summary" in report
    assert "warn1" in report


def test_exporter_saves_figures(tmp_path, simple_doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exporter import export
    from analyzers.base import AnalyzerResult

    fig, ax = plt.subplots()
    fig.set_label("test_fig")
    ax.bar(["a"], [1])
    results = [AnalyzerResult(name="test", metrics={}, figures=[fig], summary="ok")]

    export(simple_doc, results, str(tmp_path / "out"))

    assert (tmp_path / "out" / "visuals" / "test_fig.png").exists()
    assert (tmp_path / "out" / "visuals" / "test_fig.svg").exists()
