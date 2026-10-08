"""Stable detector boundary used by the API/pipeline.

Only this module knows which ChangeFormer implementation is active. Replacing
the mock with an isolated subprocess later must not change API code or schemas.
"""

from pathlib import Path

from .mock_change_detector import run_mock_change_detection


def run_change_detection(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path,
) -> dict:
    return run_mock_change_detection(before_path, after_path, output_dir)
