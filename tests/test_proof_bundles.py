"""Standalone proof bundles validate their complete artifact sets."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = sorted(ROOT.glob("analyses/**/proof/verify_equivalence.py"))


@pytest.mark.parametrize("checker", BUNDLES, ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_proof_bundle_checks_integrity_and_finite_certificate(tmp_path, checker):
    bundle = tmp_path / "proof"
    shutil.copytree(checker.parent, bundle)

    def check():
        return subprocess.run(
            [sys.executable, str(bundle / "verify_equivalence.py"), "--quick"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    result = check()
    assert result.returncode == 0, result.stdout + result.stderr
    path = bundle / "fingerprints.json"
    manifest = json.loads(path.read_text())
    assert "verify_equivalence.py" in manifest
    assert "experiment.json" not in manifest
    name = next(name for name in manifest if name.endswith(".csv"))
    artifact = bundle / name
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"\n")
    result = check()
    assert result.returncode != 0 and "checksum mismatch" in result.stderr
    artifact.write_bytes(original)
    reduced = dict(manifest)
    del reduced[name]
    path.write_text(json.dumps(reduced))
    result = check()
    assert result.returncode != 0 and "exactly the required artifacts" in result.stderr
    path.write_text(json.dumps(manifest))
    artifact.unlink()
    result = check()
    assert result.returncode != 0 and "Missing proof artifact" in result.stderr
