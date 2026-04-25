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
        nlp = doc.annotations.get("nlp")
        weights = _tfidf_weights(doc.segments, nlp)

        if not weights:
            weights = {"(keine Daten)": 1}

        top_words = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate_from_frequencies(weights)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.set_label("wordcloud")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Wortwolke (TF-IDF gewichtet)")
        fig.tight_layout()

        metrics = {"top_words": dict(top_words[:10])}
        summary = f"Top-Wörter: {', '.join(w for w, _ in top_words[:5])}"
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
