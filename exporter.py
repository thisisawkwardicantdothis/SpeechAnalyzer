import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import TranscriptDoc


def export(doc: TranscriptDoc, results: list, output_dir: str) -> None:
    base = Path(output_dir)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(exist_ok=True)
    (base / "visuals").mkdir(exist_ok=True)

    with open(base / "data" / "transcript_raw.txt", "w", encoding="utf-8") as f:
        for seg in doc.segments:
            f.write(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}\n")

    with open(base / "data" / "transcript_clean.txt", "w", encoding="utf-8") as f:
        for seg in doc.segments:
            f.write(seg.text + "\n")

    metrics_all = {result.name: result.metrics for result in results}
    with open(base / "data" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    with open(base / "reports" / "report.txt", "w", encoding="utf-8") as f:
        f.write("=== SpeechAnalyzer Report ===\n\n")
        f.write(f"Language: {doc.language}\n")
        f.write(f"Segments: {len(doc.segments)}\n\n")
        for result in results:
            f.write(f"--- {result.name} ---\n")
            f.write(result.summary + "\n")
            for warning in result.warnings:
                f.write(f"  WARNING: {warning}\n")
            f.write("\n")

    for result in results:
        for fig in result.figures:
            fig_name = fig.get_label() or result.name
            for ext in ("png", "svg"):
                fig.savefig(base / "visuals" / f"{fig_name}.{ext}", bbox_inches="tight")
            plt.close(fig)
