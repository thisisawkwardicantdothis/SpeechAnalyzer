import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

CONFIDENCE_THRESHOLD = 0.5
MIN_PAUSE = 0.2


def _tokens(spacy_doc) -> list:
    return [t for t in spacy_doc if not t.is_space and not t.is_punct]


def _wpm_net(segments: list, token_count: int) -> tuple:
    speech_seconds = sum(
        s.end - s.start for s in segments if s.confidence >= CONFIDENCE_THRESHOLD
    )
    if speech_seconds < 1e-9:
        return 0.0, 0.0
    return token_count / (speech_seconds / 60), speech_seconds


def _wpm_gross(segments: list, token_count: int) -> tuple:
    if not segments:
        return 0.0, 0.0
    duration = segments[-1].end - segments[0].start
    if duration == 0:
        return 0.0, 0.0
    return token_count / (duration / 60), duration


def _per_segment_wpm(segments: list) -> list:
    wpms = []
    for s in segments:
        duration = s.end - s.start
        if duration < 1e-9:
            continue
        words = len(s.text.split())
        wpms.append(words / (duration / 60))
    return wpms


def _detect_pauses(segments: list) -> list:
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap >= MIN_PAUSE:
            gaps.append(gap)
    return gaps


class SpeechRateAnalyzer(BaseAnalyzer):
    name = "speech_rate"
    requires_pos = False

    def run(self, doc) -> AnalyzerResult:
        token_count = len(_tokens(doc.spacy_doc))
        wpm_net, net_seconds = _wpm_net(doc.segments, token_count)
        wpm_gross, gross_seconds = _wpm_gross(doc.segments, token_count)
        seg_wpms = _per_segment_wpm(doc.segments)
        pauses = _detect_pauses(doc.segments)
        wpm_std = round(statistics.stdev(seg_wpms), 1) if len(seg_wpms) > 1 else 0.0
        total_pause = sum(pauses)
        silence_ratio = round(total_pause / gross_seconds, 3) if gross_seconds > 0 else 0.0

        metrics = {
            "wpm_net": round(wpm_net, 1),
            "wpm_gross": round(wpm_gross, 1),
            "wpm_std": wpm_std,
            "total_tokens": token_count,
            "net_speech_seconds": round(net_seconds, 1),
            "gross_duration_seconds": round(gross_seconds, 1),
            "pause_count": len(pauses),
            "total_pause_seconds": round(total_pause, 1),
            "silence_ratio": silence_ratio,
        }

        # Figure 1: Net vs Gross WPM
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.set_label("speech_rate")
        bars = ax1.bar(["Net WPM", "Gross WPM"], [wpm_net, wpm_gross], color=["#4C72B0", "#DD8452"])
        for bar, val in zip(bars, [wpm_net, wpm_gross]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=11)
        ax1.set_ylabel("Words per minute")
        ax1.set_title(f"Speech rate  |  σ={wpm_std:.1f} WPM")
        fig1.tight_layout()

        # Figure 2: WPM per segment
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.set_label("speech_rate_over_time")
        if seg_wpms:
            ax2.plot(range(1, len(seg_wpms) + 1), seg_wpms, marker="o", color="#4C72B0", linewidth=2, markersize=6)
            ax2.fill_between(range(1, len(seg_wpms) + 1), seg_wpms, alpha=0.15, color="#4C72B0")
            ax2.axhline(wpm_net, color="#C44E52", linestyle="--", label=f"Net avg {wpm_net:.0f}")
            ax2.set_xlabel("Segment")
            ax2.set_ylabel("WPM")
            ax2.set_title("Speech rate over time")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No segments", ha="center", va="center", transform=ax2.transAxes)
        fig2.tight_layout()

        # Figure 3: Pause histogram
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        fig3.set_label("speech_rate_pauses")
        if pauses:
            mean_p = total_pause / len(pauses)
            ax3.hist(pauses, bins=max(5, min(20, len(pauses))), color="#55A868", edgecolor="white")
            ax3.axvline(mean_p, color="#C44E52", linestyle="--", label=f"avg {mean_p:.1f}s")
            ax3.set_xlabel("Pause duration (s)")
            ax3.set_ylabel("Frequency")
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No pauses detected", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title(f"Pauses  |  {len(pauses)} pauses, {total_pause:.1f}s  |  Silence: {silence_ratio*100:.1f}%")
        fig3.tight_layout()

        summary = (
            f"Speech rate: net={wpm_net:.0f} WPM, gross={wpm_gross:.0f} WPM, "
            f"σ={wpm_std:.1f}, pauses={len(pauses)}, silence={silence_ratio*100:.1f}%"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig1, fig2, fig3], summary=summary)
