import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

MIN_PAUSE = 0.2


def _detect_pauses(segments: list) -> list:
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap >= MIN_PAUSE:
            pauses.append((segments[i].end, gap))
    return pauses


class PausesAnalyzer(BaseAnalyzer):
    name = "pauses"
    requires_pos = False

    def run(self, doc) -> AnalyzerResult:
        pauses = _detect_pauses(doc.segments)
        durations = [d for _, d in pauses]
        positions = [p for p, _ in pauses]

        gross_duration = (doc.segments[-1].end - doc.segments[0].start) if doc.segments else 0.0
        total_pause = sum(durations)
        silence_ratio = round(total_pause / gross_duration, 3) if gross_duration > 0 else 0.0
        mean_pause = round(statistics.mean(durations), 2) if durations else 0.0
        max_pause = round(max(durations), 2) if durations else 0.0
        std_pause = round(statistics.stdev(durations), 2) if len(durations) > 1 else 0.0

        metrics = {
            "pause_count": len(pauses),
            "total_pause_seconds": round(total_pause, 2),
            "mean_pause_duration": mean_pause,
            "max_pause_duration": max_pause,
            "std_pause_duration": std_pause,
            "silence_ratio": silence_ratio,
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.set_label("pauses")

        # Left: duration histogram
        if durations:
            axes[0].hist(durations, bins=max(5, min(20, len(durations))),
                         color="#55A868", edgecolor="white")
            axes[0].axvline(mean_pause, color="#C44E52", linestyle="--",
                            label=f"Ø {mean_pause:.1f}s")
            axes[0].set_xlabel("Pause duration (s)")
            axes[0].set_ylabel("Frequency")
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, "No pauses detected",
                         ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"Pause distribution  |  {len(pauses)} pauses, {total_pause:.1f}s total")

        # Right: pause timeline
        if pauses:
            axes[1].scatter(positions, durations, color="#55A868", s=60, alpha=0.85, zorder=3)
            axes[1].vlines(positions, 0, durations, color="#55A868", linewidth=1.5, alpha=0.45)
            axes[1].set_xlabel("Position in audio (s)")
            axes[1].set_ylabel("Duration (s)")
        else:
            axes[1].text(0.5, 0.5, "No pauses detected",
                         ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(f"Pause timeline  |  Silence: {silence_ratio*100:.1f}%")

        fig.tight_layout()

        summary = (
            f"Pauses: {len(pauses)} detected, avg {mean_pause:.1f}s, "
            f"max {max_pause:.1f}s, silence {silence_ratio*100:.1f}%"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
