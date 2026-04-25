from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from analyzers.base import BaseAnalyzer, AnalyzerResult


def _content_lemmas(spacy_doc) -> list:
    return [
        t.lemma_.lower()
        for t in spacy_doc
        if not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) > 1
    ]


def _tfidf_weights(segments: list, nlp) -> dict:
    docs = []
    for seg in segments:
        lemmas = [
            t.lemma_.lower()
            for t in nlp(seg.text)
            if not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) > 1
        ]
        if lemmas:
            docs.append(" ".join(lemmas))

    if len(docs) < 2:
        counts = Counter(" ".join(docs).split()) if docs else Counter()
        total = sum(counts.values()) or 1
        return {w: c / total for w, c in counts.items()}

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)
    scores = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    return dict(zip(words, scores.tolist()))


class WordcloudAnalyzer(BaseAnalyzer):
    name = "wordcloud"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        nlp = doc.annotations["nlp"]
        weights = _tfidf_weights(doc.segments, nlp)

        if not weights:
            weights = {"(no data)": 1}

        top_words = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate_from_frequencies(weights)

        # Figure 1: Word cloud
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        fig1.set_label("wordcloud")
        ax1.imshow(wc, interpolation="bilinear")
        ax1.axis("off")
        ax1.set_title("Word cloud (TF-IDF weighted)")
        fig1.tight_layout()

        # Figure 2: Top-20 bar chart
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        fig2.set_label("wordcloud_top20")
        if top_words:
            words = [w for w, _ in reversed(top_words)]
            scores = [s for _, s in reversed(top_words)]
            colors = [plt.cm.viridis(i / max(len(words) - 1, 1)) for i in range(len(words))]
            bars = ax2.barh(words, scores, color=colors)
            for bar, score in zip(bars, scores):
                ax2.text(
                    bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=8,
                )
            ax2.set_xlabel("TF-IDF Score")
            ax2.set_title("Top-20 words by TF-IDF")
            ax2.margins(x=0.15)
        fig2.tight_layout()

        metrics = {"top_words": dict(top_words[:10])}
        summary = f"Top words: {', '.join(w for w, _ in top_words[:5])}"
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig1, fig2], summary=summary)
