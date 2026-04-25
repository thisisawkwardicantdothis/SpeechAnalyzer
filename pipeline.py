from analyzers import ANALYZER_REGISTRY
from analyzers.base import AnalyzerResult
from preprocessor import preprocess
from exporter import export


def run_pipeline(segments: list, language: str, modules: list, output_dir: str) -> list:
    doc = preprocess(segments, language)

    results = []
    for module_name in modules:
        analyzer = ANALYZER_REGISTRY.get(module_name)
        if analyzer is None:
            continue
        if not analyzer.can_run(doc):
            results.append(AnalyzerResult(
                name=module_name,
                metrics={},
                figures=[],
                summary=f"{module_name} not available for language '{doc.language}'",
                warnings=[f"Language '{doc.language}' not in SUPPORTED_LANGUAGES — module skipped"],
            ))
            continue
        results.append(analyzer.run(doc))

    export(doc, results, output_dir)
    return results
