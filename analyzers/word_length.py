import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult


def _content_word_lengths(spacy_doc) -> list:
    return [
        len(t.text)
        for t in spacy_doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.text.strip()
    ]


class WordLengthAnalyzer(BaseAnalyzer):
    name = "word_length"
    requires_pos = False

    def run(self, doc) -> AnalyzerResult:
        lengths = _content_word_lengths(doc.spacy_doc)
        word_count = len(lengths)
        if not lengths:
            lengths = [0]

        mean_len = statistics.mean(lengths)
        median_len = statistics.median(lengths)
        max_len = max(lengths)
        std_len = round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0.0
        sorted_l = sorted(lengths)
        p25 = sorted_l[len(sorted_l) // 4]
        p75 = sorted_l[min(3 * len(sorted_l) // 4, len(sorted_l) - 1)]

        metrics = {
            "mean_word_length": round(mean_len, 2),
            "median_word_length": float(median_len),
            "max_word_length": max_len,
            "std_word_length": std_len,
            "p25_word_length": p25,
            "p75_word_length": p75,
            "word_count": word_count,
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.set_label("word_length_dist")

        # Left: histogram
        axes[0].hist(lengths, bins=range(1, max_len + 2), edgecolor="white", color="#4C72B0", align="left")
        axes[0].axvline(mean_len, color="#C44E52", linestyle="--", label=f"Ø {mean_len:.1f}")
        axes[0].set_xlabel("Word length (chars)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Word length distribution")
        axes[0].legend()

        # Right: boxplot
        axes[1].boxplot(
            lengths, patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", alpha=0.7),
            medianprops=dict(color="#C44E52", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        axes[1].set_ylabel("Word length (chars)")
        axes[1].set_title(f"Boxplot  |  σ={std_len:.1f}, P25={p25}, P75={p75}")
        axes[1].set_xticks([])

        fig.tight_layout()

        summary = (
            f"Word length: avg={mean_len:.1f}, median={median_len:.0f}, "
            f"σ={std_len:.1f}, P25={p25}, P75={p75}, max={max_len}"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
