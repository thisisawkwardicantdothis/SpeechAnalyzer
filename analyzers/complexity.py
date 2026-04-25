import math
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
_POS_LABELS = {
    "NOUN": "Noun", "VERB": "Verb", "ADJ": "Adjective", "ADV": "Adverb",
    "PRON": "Pronoun", "DET": "Article", "ADP": "Preposition",
    "CCONJ": "Conjunction", "SCONJ": "Conjunction", "PART": "Particle",
    "NUM": "Numeral", "PROPN": "Proper noun", "INTJ": "Interjection",
}


def _get_lemmas(spacy_doc):
    return [
        t.lemma_.lower()
        for t in spacy_doc
        if not t.is_punct and not t.is_space and t.lemma_.strip()
    ]


def _brunet(tokens: list, lemmas: list) -> float:
    n = len(tokens)
    v = len(set(lemmas))
    if n == 0 or v == 0:
        return 0.0
    return n ** (v ** -0.165)


def _honore(tokens: list, lemmas: list) -> float:
    n = len(tokens)
    v = len(set(lemmas))
    if n == 0 or v == 0:
        return 0.0
    counts = Counter(lemmas)
    f1 = sum(1 for c in counts.values() if c == 1)
    if f1 == v:
        return 0.0
    return 100 * math.log(n) / (1 - f1 / v)


def _lexical_density(spacy_doc) -> float:
    tokens = [t for t in spacy_doc if not t.is_space and not t.is_punct]
    if not tokens:
        return 0.0
    content = sum(1 for t in tokens if t.pos_ in CONTENT_POS)
    return content / len(tokens)


def _sentence_stats(spacy_doc) -> tuple:
    lengths = [
        len([t for t in sent if not t.is_space and not t.is_punct])
        for sent in spacy_doc.sents
    ]
    if not lengths:
        return 0, 0.0, 0.0
    avg = statistics.mean(lengths)
    std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    return len(lengths), round(avg, 2), round(std, 2)


def _pos_distribution(spacy_doc) -> dict:
    tokens = [t for t in spacy_doc if not t.is_space and not t.is_punct]
    counts = Counter(t.pos_ for t in tokens)
    total = len(tokens) or 1
    return {pos: round(count / total * 100, 1) for pos, count in counts.most_common()}


class ComplexityAnalyzer(BaseAnalyzer):
    name = "complexity"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        tokens_all = [t for t in doc.spacy_doc if not t.is_space and not t.is_punct]
        lemmas = _get_lemmas(doc.spacy_doc)
        token_texts = [t.text.lower() for t in tokens_all]

        brunet = _brunet(token_texts, lemmas)
        honore = _honore(token_texts, lemmas)
        lex_density = _lexical_density(doc.spacy_doc)
        sent_count, avg_sent_len, sent_std = _sentence_stats(doc.spacy_doc)
        pos_dist = _pos_distribution(doc.spacy_doc)

        metrics = {
            "brunet_index": round(brunet, 4),
            "honore_index": round(honore, 4),
            "lexical_density": round(lex_density, 4),
            "sentence_count": sent_count,
            "avg_sentence_length": avg_sent_len,
            "sentence_length_std": sent_std,
            "pos_distribution": pos_dist,
        }

        # Figure 1: Brunet / Honore / Lex Density
        fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig1.set_label("complexity_metrics")
        axes[0].bar(["Brunet Index"], [brunet], color="#4C72B0")
        axes[0].set_title("Brunet Index")
        axes[0].set_ylabel("Value")
        axes[1].bar(["Honoré Index"], [honore], color="#DD8452")
        axes[1].set_title("Honoré Index")
        axes[1].set_ylabel("Value")
        axes[2].bar(["Lex. Density"], [lex_density], color="#55A868")
        axes[2].set_ylim(0, 1)
        axes[2].set_title(f"Lex. Density  |  {sent_count} sentences, avg {avg_sent_len:.1f} w.")
        axes[2].set_ylabel("Proportion")
        fig1.tight_layout()

        # Figure 2: POS distribution
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.set_label("pos_distribution")
        if pos_dist:
            threshold = 3.0
            main_labels, main_vals, other = [], [], 0.0
            for pos, pct in pos_dist.items():
                label = _POS_LABELS.get(pos, pos)
                if pct >= threshold:
                    main_labels.append(f"{label}\n{pct:.1f}%")
                    main_vals.append(pct)
                else:
                    other += pct
            if other > 0:
                main_labels.append(f"Other\n{other:.1f}%")
                main_vals.append(other)
            ax2.pie(
                main_vals, labels=main_labels, startangle=90,
                colors=list(plt.cm.Set3.colors[: len(main_vals)]),
                wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
            )
            ax2.set_title("Part-of-speech distribution")
        fig2.tight_layout()

        summary = (
            f"Brunet={brunet:.2f}, Honoré={honore:.1f}, "
            f"Lex. Density={lex_density:.3f}, {sent_count} sentences, avg {avg_sent_len:.1f} w./sent"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig1, fig2], summary=summary)
