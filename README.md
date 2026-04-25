# SpeechAnalyzer

A local CLI tool that transcribes video files and runs linguistic analysis on the spoken content — fully offline, no API keys required.

Transcribes audio with [OpenAI Whisper](https://github.com/openai/whisper), processes the transcript with [spaCy](https://spacy.io/), runs a suite of linguistic analyzers, and exports structured reports, metrics, and charts.

---

## What it analyzes

| Module | What it measures |
|--------|-----------------|
| `vocabulary` | Type-token ratio (TTR), MATTR, Chao1 vocabulary estimate, hapax legomena, top-10 content words |
| `complexity` | Brunet index, Honoré index, lexical density, sentence length statistics, part-of-speech distribution |
| `speech_rate` | Net and gross words-per-minute, per-segment WPM, pause detection, silence ratio |
| `word_length` | Mean, median, standard deviation, and distribution of word lengths in characters |
| `wordcloud` | TF-IDF weighted word cloud and top-20 word bar chart |
| `sentences` | Sentence count, average and median length, length distribution over time |
| `pauses` | Pause count, duration statistics, silence ratio, pause timeline |

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) — must be on your `PATH`
- ~500 MB disk space for Whisper base model + spaCy language models (auto-downloaded on first run)

---

## Installation

```bash
git clone https://github.com/your-username/SpeechAnalyzer.git
cd SpeechAnalyzer

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run will automatically download the spaCy model for the detected language. An internet connection is required for that initial download only.

---

## Usage

```bash
python analyze.py <path/to/video.mp4>
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--modules` | `all` | Comma-separated list of modules to run, or `all` |
| `--output` | `output/` | Base directory for results |
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |

### Examples

```bash
# Run all modules with the default Whisper model
python analyze.py lecture.mp4

# Run only vocabulary and speech rate
python analyze.py lecture.mp4 --modules vocabulary,speech_rate

# Use a more accurate (but slower) model, custom output folder
python analyze.py interview.mkv --whisper-model medium --output ./results
```

---

## Output structure

```
output/<video_name>/
  data/
    transcript_raw.txt    # timestamped Whisper segments
    transcript_clean.txt  # deduplicated transcript text
    metrics.json          # all numeric results as JSON
  reports/
    report.txt            # human-readable summary with any warnings
  visuals/
    *.png / *.svg         # one or more charts per module
```

---

## Supported languages

Automatic spaCy model selection for: **German, English, French, Spanish, Italian, Dutch, Portuguese**. For any other language detected by Whisper, a multilingual fallback model is used and POS-dependent analyzers are skipped with a warning.

### Whisper model sizes

| Model | Size | Notes |
|-------|------|-------|
| `tiny` | ~75 MB | fastest, lowest accuracy |
| `base` | ~142 MB | good balance (default) |
| `small` | ~461 MB | better accuracy |
| `medium` | ~1.5 GB | high accuracy |
| `large` | ~2.9 GB | highest accuracy, slowest |

---

## Running the tests

```bash
pytest
```

The test suite downloads `en_core_web_sm` automatically on first run and does not require a real video file.

---

## Project structure

```
SpeechAnalyzer/
  analyze.py          # CLI entry point and ffmpeg check
  pipeline.py         # orchestrates transcribe → preprocess → analyse → export
  transcriber.py      # Whisper wrapper → list[Segment] + detected language
  preprocessor.py     # segment deduplication and spaCy model loading
  exporter.py         # writes output/ folder
  analyzers/
    base.py           # Segment, TranscriptDoc, AnalyzerResult, BaseAnalyzer
    __init__.py       # ANALYZER_REGISTRY
    vocabulary.py
    complexity.py
    speech_rate.py
    word_length.py
    wordcloud_gen.py
    sentences.py
    pauses.py
  tests/
    conftest.py
    test_vocabulary.py
    test_complexity.py
    test_speech_rate.py
    test_pauses.py
    test_sentences.py
```

---

## Adding a custom analyzer

1. Create `analyzers/my_module.py` extending `BaseAnalyzer`:

```python
from analyzers.base import BaseAnalyzer, AnalyzerResult

class MyAnalyzer(BaseAnalyzer):
    name = "my_module"
    requires_pos = False  # True if you need spaCy POS tags

    def run(self, doc) -> AnalyzerResult:
        metrics = {"example_metric": 42}
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[], summary="example: 42")
```

2. Register it in `analyzers/__init__.py`:

```python
from analyzers.my_module import MyAnalyzer
ANALYZER_REGISTRY["my_module"] = MyAnalyzer()
```

---

## License

MIT
