"""
oader.py - 数据加载层（高层 API）
这个文件提供了方便的工具函数来加载和管理数据集。
"""
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
    定义 DocVQA 数据集的标准目录结构
    设计目的：集中管理路径，避免硬编码散落在各处
    """

    root: Path
    qas_dir: Path
    images_dir: Path
    ocr_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "DocVQAPaths":
        """
        工厂方法：只需提供根目录，自动推导其他路径
        预期目录结构：
        root/
        ├── spdocvqa_qas/    # QA 标注 JSON
        ├── spdocvqa_images/ # 图像文件
        └── spdocvqa_ocr/    # OCR JSON
        """
        root = Path(root)
        return cls(
            root=root,
            qas_dir=root / "spdocvqa_qas",
            images_dir=root / "spdocvqa_images",
            ocr_dir=root / "spdocvqa_ocr",
        )


# 基础加载函数
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


# 图像路径处理
def resolve_docvqa_image_path(images_dir: str | Path, relative_image_path: str) -> str:
    """
    Resolve a relative image path like:
    documents/xnbl0037_1.png

    into:
    /.../spdocvqa_images/documents/xnbl0037_1.png
    """
    images_dir = Path(images_dir)
    return str(images_dir / relative_image_path)


def resolve_docvqa_ocr_path(ocr_dir: str | Path, relative_image_path: str) -> str:
    """
    Resolve OCR path from relative image path.

    Example:
    documents/xnbl0037_1.png
    ->
    /.../spdocvqa_ocr/xnbl0037_1.json

    Assumption:
    OCR file name and image file name are one-to-one matched by stem.
    """
    ocr_dir = Path(ocr_dir)
    image_name = Path(relative_image_path).name
    ocr_name = Path(image_name).stem + ".json"
    return str(ocr_dir / ocr_name)


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


def attach_ocr_paths(
        manifest: DatasetManifest,
        ocr_dir: str | Path,
) -> DatasetManifest:
    """
    Attach OCR path to sample.meta["ocr_path"] using image file stem.
    """
    for sample in manifest.samples:
        sample.meta["ocr_path"] = resolve_docvqa_ocr_path(ocr_dir, sample.image_path)
    return manifest


def load_docvqa_manifest_with_assets(
    qas_json_path: str | Path,
    images_dir: str | Path,
    ocr_dir: str | Path,
) -> DatasetManifest:
    """
    Load a DocVQA manifest, convert image paths to absolute paths,
    and attach OCR path into sample.meta["ocr_path"].
    """
    manifest = load_docvqa_manifest(qas_json_path)
    manifest = attach_absolute_image_paths(manifest, images_dir)
    manifest = attach_ocr_paths(manifest, ocr_dir)
    return manifest


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