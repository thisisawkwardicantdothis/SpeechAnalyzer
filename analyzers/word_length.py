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
        if not lengths:
            lengths = [0]

        mean_len = statistics.mean(lengths)
        median_len = statistics.median(lengths)
        max_len = max(lengths)

        metrics = {
            "mean_word_length": round(mean_len, 2),
            "median_word_length": float(median_len),
            "max_word_length": max_len,
            "word_count": len(lengths),
        }

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.set_label("word_length_dist")
        ax.hist(lengths, bins=range(1, max_len + 2), edgecolor="white", color="#4C72B0", align="left")
        ax.axvline(mean_len, color="#C44E52", linestyle="--", label=f"Ø {mean_len:.1f}")
        ax.set_xlabel("Wortlänge (Zeichen)")
        ax.set_ylabel("Häufigkeit")
        ax.set_title("Wortlängenverteilung")
        ax.legend()
        fig.tight_layout()

        summary = f"Wortlänge: Ø={mean_len:.1f}, Median={median_len:.0f}, Max={max_len} Zeichen"
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
