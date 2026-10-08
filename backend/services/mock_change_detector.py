from pathlib import Path


def run_mock_change_detection(
    before_path: str | Path,
    after_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Return a deterministic result matching the future ChangeFormer wrapper.

    The image paths are intentionally accepted but not read. This keeps the mock
    independent of raster/model dependencies while preserving the subprocess
    contract that the real detector will implement later.
    """
    del before_path, after_path
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / "mock_change_mask.pgm"
    width = height = 512
    changed_pixels = 24892
    total_pixels = width * height
    with mask_path.open("wb") as mask:
        mask.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        mask.write(bytes([255 if i < changed_pixels else 0 for i in range(total_pixels)]))
    return {
        "success": True,
        "mask_path": str(mask_path),
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": round(changed_pixels / total_pixels * 100, 2),
    }
