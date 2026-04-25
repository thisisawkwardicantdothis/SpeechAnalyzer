import matplotlib.pyplot as plt
import pytest
import spacy


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")

@pytest.fixture(scope="session")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

@pytest.fixture
def simple_segments():
    from analyzers.base import Segment
    return [
        Segment(start=0.0, end=4.0, text="I love programming and coding every day", confidence=0.9),
        Segment(start=4.0, end=8.0, text="Python is my favorite programming language", confidence=0.85),
        Segment(start=9.0, end=13.0, text="I enjoy writing code and learning new things", confidence=0.8),
    ]

@pytest.fixture
def simple_doc(nlp, simple_segments):
    from analyzers.base import TranscriptDoc
    text = " ".join(s.text for s in simple_segments)
    spacy_doc = nlp(text)
    return TranscriptDoc(
        raw_text=text,
        clean_text=text,
        segments=simple_segments,
        spacy_doc=spacy_doc,
        language="en",
        annotations={"nlp": nlp},
    )
