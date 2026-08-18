import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs" / "imagenet1k_val" / "causal_followups"
FIGURES = ROOT / "results" / "figures"


def test_manifest_references_valid_run_files():
    manifest = json.loads((RUNS / "manifest.json").read_text())
    referenced = []
    for group in manifest["groups"].values():
        for filename in group["files"]:
            path = RUNS / filename
            assert path.is_file(), path
            json.loads(path.read_text())
            referenced.append(filename)

    assert len(referenced) == 26
    assert len(referenced) == len(set(referenced))


def test_manifest_references_renderable_figures():
    manifest = json.loads((RUNS / "manifest.json").read_text())
    for filename in manifest["figures"]:
        path = FIGURES / filename
        assert path.is_file(), path
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_results_document_links_to_existing_figures():
    document = (ROOT / "docs" / "CAUSAL_RESULTS.md").read_text()
    links = re.findall(r"\]\(\.\./results/figures/([^\)]+\.png)\)", document)
    assert len(links) == 10
    assert len(links) == len(set(links))
    for filename in links:
        assert (FIGURES / filename).is_file(), filename


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        test()
    print(f"All {len(tests)} tests passed.")
