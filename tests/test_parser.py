import json
from pathlib import Path

from mm_docvqa.data.loader import (
    attach_absolute_image_paths,
    get_default_docvqa_qas_file,
    load_docvqa_manifest,
    load_docvqa_ocr_page,
    resolve_docvqa_image_path,
)
from mm_docvqa.data.parser_docvqa import (
    parse_docvqa_manifest,
    parse_docvqa_ocr_page,
    parse_docvqa_sample,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_docvqa_sample() -> None:
    raw_sample = json.loads((FIXTURE_DIR / "fake_docvqa_sample.json").read_text(encoding="utf-8"))

    sample = parse_docvqa_sample(raw_sample, default_split="train")

    assert sample.dataset_name == "docvqa"
    assert sample.split == "train"
    assert sample.question_id == "337"
    assert sample.question == "what is the date mentioned in this letter?"
    assert sample.image_path == "documents/xnbl0037_1.png"
    assert sample.answers == ["1/8/93"]
    assert sample.question_types == ["handwritten", "form"]
    assert sample.doc_numeric_id == 279
    assert sample.doc_id == "xnbl0037"
    assert sample.page_no == 1


def test_parse_docvqa_manifest() -> None:
    raw_sample = json.loads((FIXTURE_DIR / "fake_docvqa_sample.json").read_text(encoding="utf-8"))

    raw_manifest = {
        "dataset_name": "SP-DocVQA",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [raw_sample],
    }

    manifest = parse_docvqa_manifest(raw_manifest)

    assert manifest.dataset_name == "docvqa"
    assert manifest.version == "1.0"
    assert manifest.split == "train"
    assert len(manifest.samples) == 1
    assert manifest.samples[0].question_id == "337"


def test_parse_docvqa_ocr_page() -> None:
    raw_ocr = json.loads((FIXTURE_DIR / "fake_docvqa_ocr.json").read_text(encoding="utf-8"))

    page = parse_docvqa_ocr_page(raw_ocr)

    assert page.page_no == 1
    assert page.width == 1692
    assert page.height == 2245
    assert page.unit == "pixel"
    assert page.status == "Succeeded"
    assert page.source == "microsoft_ocr"
    assert page.n_lines >= 1
    assert "REYNOLDS" in page.full_text()


def test_load_docvqa_manifest(tmp_path: Path) -> None:
    raw_sample = json.loads((FIXTURE_DIR / "fake_docvqa_sample.json").read_text(encoding="utf-8"))

    qas_path = tmp_path / "train_v1.0_withQT.json"
    qas_path.write_text(
        json.dumps(
            {
                "dataset_name": "SP-DocVQA",
                "dataset_version": "1.0",
                "dataset_split": "train",
                "data": [raw_sample],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_docvqa_manifest(qas_path)

    assert manifest.split == "train"
    assert len(manifest.samples) == 1
    assert manifest.samples[0].question == "what is the date mentioned in this letter?"


def test_load_docvqa_ocr_page(tmp_path: Path) -> None:
    raw_ocr = json.loads((FIXTURE_DIR / "fake_docvqa_ocr.json").read_text(encoding="utf-8"))

    ocr_path = tmp_path / "fake_ocr.json"
    ocr_path.write_text(json.dumps(raw_ocr), encoding="utf-8")

    page = load_docvqa_ocr_page(ocr_path)

    assert page.page_no == 1
    assert page.n_lines >= 1


def test_resolve_docvqa_image_path() -> None:
    images_dir = "/datasets/spdocvqa_images"
    relative_image_path = "/ffbf0227_1.png"

    resolved = resolve_docvqa_image_path(images_dir, relative_image_path)

    assert resolved.endswith("/datasets/spdocvqa_images/ffbf0227_1.png")


def test_attach_absolute_image_paths() -> None:
    raw_sample = json.loads((FIXTURE_DIR / "fake_docvqa_sample.json").read_text(encoding="utf-8"))

    raw_manifest = {
        "dataset_name": "SP-DocVQA",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [raw_sample],
    }

    manifest = parse_docvqa_manifest(raw_manifest)
    manifest = attach_absolute_image_paths(manifest, "/tmp/spdocvqa_images")

    assert manifest.samples[0].image_path.endswith("spdocvqa_images/documents/xnbl0037_1.png")


def test_get_default_docvqa_qas_file(tmp_path: Path) -> None:
    qas_dir = tmp_path / "spdocvqa_qas"
    qas_dir.mkdir()

    train_path = get_default_docvqa_qas_file(qas_dir, "train")
    val_path = get_default_docvqa_qas_file(qas_dir, "val")
    test_path = get_default_docvqa_qas_file(qas_dir, "test")

    assert train_path.name == "train_v1.0_withQT.json"
    assert val_path.name == "val_v1.0_withQT.json"
    assert test_path.name == "test_v1.0.json"
