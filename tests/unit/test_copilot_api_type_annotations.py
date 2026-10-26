"""Regression tests for importable Copilot API type annotations."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


COPILOT_API_DIR = Path(__file__).resolve().parents[2] / "apps" / "copilot-api"


@pytest.mark.parametrize(
    ("filename", "expected_symbol"),
    [("config.py", "Settings"), ("schemas.py", "QueryRequest")],
)
def test_copilot_api_modules_import_with_standard_type_annotations(
    filename: str, expected_symbol: str
) -> None:
    module_path = COPILOT_API_DIR / filename
    spec = spec_from_file_location(f"copilot_api_{module_path.stem}", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, expected_symbol)
