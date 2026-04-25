from analyzers.vocabulary import VocabularyAnalyzer
from analyzers.complexity import ComplexityAnalyzer
from analyzers.speech_rate import SpeechRateAnalyzer
from analyzers.word_length import WordLengthAnalyzer
from analyzers.wordcloud_gen import WordcloudAnalyzer
from analyzers.sentences import SentencesAnalyzer
from analyzers.pauses import PausesAnalyzer

ANALYZER_REGISTRY: dict = {
    "vocabulary": VocabularyAnalyzer(),
    "complexity": ComplexityAnalyzer(),
    "speech_rate": SpeechRateAnalyzer(),
    "word_length": WordLengthAnalyzer(),
    "wordcloud": WordcloudAnalyzer(),
    "sentences": SentencesAnalyzer(),
    "pauses": PausesAnalyzer(),
}

ALL_MODULES: list = list(ANALYZER_REGISTRY.keys())
