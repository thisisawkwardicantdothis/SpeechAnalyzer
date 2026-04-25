from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult


def _get_lemmas(spacy_doc):
    return [
        t.lemma_.lower()
        for t in spacy_doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.lemma_.strip()
    ]


def _ttr(lemmas: list) -> float:
    if not lemmas:
        return 0.0
    return len(set(lemmas)) / len(lemmas)


def _mattr(lemmas: list, window_size: int = 50) -> float:
    if not lemmas:
        return 0.0
    if len(lemmas) <= window_size:
        return len(set(lemmas)) / len(lemmas)
    ttrs = [
        len(set(lemmas[i : i + window_size])) / window_size
        for i in range(len(lemmas) - window_size + 1)
    ]
    return sum(ttrs) / len(ttrs)


def _chao1(lemmas: list) -> tuple:
    if not lemmas:
        return 0.0, 0, 0, []
    counts = Counter(lemmas)
    s_obs = len(counts)
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    warnings = []
    if f2 == 0:
        f2 = 1
        warnings.append("Chao1: f2=0, smoothing applied (f2=1)")
    estimate = s_obs + (f1 ** 2) / (2 * f2)
    return estimate, f1, s_obs, warnings


def _growth_curve(lemmas: list) -> list:
    seen = set()
    curve = []
    for lemma in lemmas:
        seen.add(lemma)
        curve.append(len(seen))
    return curve


class VocabularyAnalyzer(BaseAnalyzer):
    name = "vocabulary"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        lemmas = _get_lemmas(doc.spacy_doc)
        ttr = _ttr(lemmas)
        mattr = _mattr(lemmas)
        chao1, f1, s_obs, warns = _chao1(lemmas)
        hapax_percent = round(f1 / len(lemmas) * 100, 2) if lemmas else 0.0
        top_words = dict(Counter(lemmas).most_common(10))

        metrics = {
            "ttr": round(ttr, 4),
            "mattr": round(mattr, 4),
            "chao1": round(chao1, 2),
            "observed_types": s_obs,
            "hapax_legomena": f1,
            "hapax_percent": hapax_percent,
            "total_lemmas": len(lemmas),
            "top_10_words": top_words,
        }

        # Figure 1: TTR/MATTR + Chao1
        fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig1.set_label("vocabulary_metrics")
        axes[0].bar(["TTR", "MATTR"], [ttr, mattr], color=["#4C72B0", "#DD8452"])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Type-Token-Ratio")
        axes[0].set_ylabel("Value")
        axes[1].bar(["Observed", "Chao1 estimate"], [s_obs, chao1], color=["#4C72B0", "#DD8452"])
        axes[1].set_title(f"Vocabulary estimate  |  Hapax: {hapax_percent:.1f}%")
        axes[1].set_ylabel("Type count")
        fig1.tight_layout()

        # Figure 2: Vocabulary growth curve
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        fig2.set_label("vocabulary_growth")
        if lemmas:
            curve = _growth_curve(lemmas)
            xs = range(1, len(curve) + 1)
            ax2.plot(xs, curve, color="#4C72B0", linewidth=2)
            ax2.fill_between(xs, curve, alpha=0.15, color="#4C72B0")
            ax2.set_xlabel("Token-Position")
            ax2.set_ylabel("Cumulative unique lemmas")
            ax2.set_title("Vocabulary growth curve")
        fig2.tight_layout()

        # Figure 3: Top-10 horizontal bar
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        fig3.set_label("vocabulary_top_words")
        if top_words:
            words = list(top_words.keys())[::-1]
            counts = list(top_words.values())[::-1]
            colors = [plt.cm.viridis(i / max(len(words) - 1, 1)) for i in range(len(words))]
            ax3.barh(words, counts, color=colors)
            ax3.set_xlabel("Frequency")
            ax3.set_title("Top-10 content words")
        fig3.tight_layout()

        summary = (
            f"Vocabulary: MATTR={mattr:.3f}, TTR={ttr:.3f}, "
            f"Chao1={chao1:.0f} (observed: {s_obs}), hapax={hapax_percent:.1f}%"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig1, fig2, fig3], summary=summary, warnings=warns)
