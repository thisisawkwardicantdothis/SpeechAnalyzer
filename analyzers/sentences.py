import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult


def _sentence_lengths(spacy_doc) -> list:
    return [
        len([t for t in sent if not t.is_space and not t.is_punct])
        for sent in spacy_doc.sents
    ]


class SentencesAnalyzer(BaseAnalyzer):
    name = "sentences"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        lengths = _sentence_lengths(doc.spacy_doc)
        if not lengths:
            lengths = [0]

        count = len(lengths)
        avg = round(statistics.mean(lengths), 2)
        median = statistics.median(lengths)
        max_len = max(lengths)
        std = round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0.0

        metrics = {
            "sentence_count": count,
            "avg_sentence_length_words": avg,
            "median_sentence_length": float(median),
            "max_sentence_length": max_len,
            "sentence_length_std": std,
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.set_label("sentences")

        # Left: histogram
        bins = max(1, min(15, max_len))
        axes[0].hist(lengths, bins=bins, color="#4C72B0", edgecolor="white")
        axes[0].axvline(avg, color="#C44E52", linestyle="--", label=f"Ø {avg:.1f}")
        axes[0].set_xlabel("Sentence length (words)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Sentence length distribution")
        axes[0].legend()

        # Right: sentence length over index
        axes[1].plot(range(1, count + 1), lengths, marker="o", color="#4C72B0",
                     linewidth=1.5, markersize=4)
        axes[1].fill_between(range(1, count + 1), lengths, alpha=0.12, color="#4C72B0")
        axes[1].axhline(avg, color="#C44E52", linestyle="--", label=f"Ø {avg:.1f}")
        axes[1].set_xlabel("Sentence #")
        axes[1].set_ylabel("Words")
        axes[1].set_title(f"Sentence length over time  |  σ={std:.1f}")
        axes[1].legend()

        fig.tight_layout()

        summary = f"Sentences: {count}, avg {avg:.1f} words, median={median:.0f}, σ={std:.1f}, max={max_len}"
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
