from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mm_docvqa.data.parser_docvqa import (
    parse_docvqa_manifest,
    parse_docvqa_ocr_page,
)
from mm_docvqa.domain.schemas import DatasetManifest, OCRPage


@dataclass(slots=True, frozen=True)
class DocVQAPaths:
    """
    Standard local paths for the downloaded DocVQA dataset.
    """

    root: Path
    qas_dir: Path
    images_dir: Path
    ocr_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DocVQAPaths":
        root = Path(root)
        return cls(
            root=root,
            qas_dir=root / "spdocvqa_qas",
            images_dir=root / "spdocvqa_images",
            ocr_dir=root / "spdocvqa_ocr",
        )


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file from disk.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_docvqa_manifest(qas_json_path: str | Path) -> DatasetManifest:
    """
    Load one DocVQA annotation JSON file and parse it into DatasetManifest.
    """
    raw_manifest = load_json(qas_json_path)
    manifest = parse_docvqa_manifest(raw_manifest)
    return manifest


def resolve_docvqa_image_path(images_dir: str | Path, relative_image_path: str) -> str:
    """
    Resolve a relative image path like:
    documents/xnbl0037_1.png

    into:
    /.../spdocvqa_images/documents/xnbl0037_1.png
    """
    images_dir = Path(images_dir)
    return str(images_dir / relative_image_path)


def attach_absolute_image_paths(
    manifest: DatasetManifest,
    images_dir: str | Path,
) -> DatasetManifest:
    """
    Mutate samples in-place so sample.image_path becomes an absolute path.
    """
    images_dir = Path(images_dir)
    for sample in manifest.samples:
        sample.image_path = str(images_dir / sample.image_path)
    return manifest


def load_docvqa_manifest_with_images(
    qas_json_path: str | Path,
    images_dir: str | Path,
) -> DatasetManifest:
    """
    Load a DocVQA manifest and convert all relative image paths to absolute paths.
    """
    manifest = load_docvqa_manifest(qas_json_path)
    return attach_absolute_image_paths(manifest, images_dir)


def load_docvqa_ocr_page(ocr_json_path: str | Path) -> OCRPage:
    """
    Load one raw OCR JSON file and parse it into OCRPage.
    """
    raw_ocr = load_json(ocr_json_path)
    return parse_docvqa_ocr_page(raw_ocr)


def get_default_docvqa_qas_file(qas_dir: str | Path, split: str) -> Path:
    """
    Return the default annotation file for a split.

    Expected file names in your current dataset:
    - train -> train_v1.0_withQT.json
    - val   -> val_v1.0_withQT.json
    - test  -> test_v1.0.json
    """
    qas_dir = Path(qas_dir)

    mapping = {
        "train": qas_dir / "train_v1.0_withQT.json",
        "val": qas_dir / "val_v1.0_withQT.json",
        "test": qas_dir / "test_v1.0.json",
    }

    if split not in mapping:
        raise ValueError(f"Unsupported split: {split}")

    return mapping[split]