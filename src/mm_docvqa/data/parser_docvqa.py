from __future__ import annotations

from typing import Any

from mm_docvqa.domain.schemas import (
    BBox,
    DatasetManifest,
    DocumentSample,
    OCRLine,
    OCRPage,
    OCRWord,
)


def _to_int(value: Any) -> int | None:
    """Convert value to int if possible, otherwise return None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return int(value)
    return int(value)


def parse_docvqa_sample(
    raw_sample: dict[str, Any],
    default_split: str | None = None,
) -> DocumentSample:
    """
    Parse one raw DocVQA sample into the internal DocumentSample.

    Raw DocVQA sample example:
    {
        "questionId": 337,
        "question": "what is the date mentioned in this letter?",
        "question_types": ["handwritten", "form"],
        "image": "documents/xnbl0037_1.png",
        "docId": 279,
        "ucsf_document_id": "xnbl0037",
        "ucsf_document_page_no": "1",
        "answers": ["1/8/93"],
        "data_split": "train"
    }
    """
    split = raw_sample.get("data_split") or default_split or "unknown"

    return DocumentSample(
        dataset_name="docvqa",
        split=split,
        question_id=str(raw_sample["questionId"]),
        question=raw_sample["question"],
        image_path=raw_sample["image"],
        answers=raw_sample.get("answers", []),
        question_types=raw_sample.get("question_types", []),
        doc_numeric_id=_to_int(raw_sample.get("docId")),
        doc_id=raw_sample.get("ucsf_document_id"),
        page_no=_to_int(raw_sample.get("ucsf_document_page_no")),
        meta={
            "raw_data_split": raw_sample.get("data_split"),
        },
    )


def parse_docvqa_manifest(raw_manifest: dict[str, Any]) -> DatasetManifest:
    """
    Parse a raw DocVQA manifest JSON into DatasetManifest.

    Raw manifest example:
    {
        "dataset_name": "SP-DocVQA",
        "dataset_version": "1.0",
        "dataset_split": "train",
        "data": [...]
    }
    """
    dataset_split = raw_manifest.get("dataset_split", "unknown")
    samples = [
        parse_docvqa_sample(sample, default_split=dataset_split)
        for sample in raw_manifest.get("data", [])
    ]

    return DatasetManifest(
        dataset_name="docvqa",
        version=str(raw_manifest.get("dataset_version", "unknown")),
        split=dataset_split,
        samples=samples,
        source_files={},
    )


def parse_docvqa_ocr_word(raw_word: dict[str, Any], index: int | None = None) -> OCRWord:
    """
    Parse one raw OCR word.
    """
    polygon = raw_word["boundingBox"]
    return OCRWord(
        text=raw_word["text"],
        bbox=BBox.from_polygon(polygon),
        index=index,
        raw_polygon=polygon,
    )


def parse_docvqa_ocr_line(raw_line: dict[str, Any], index: int | None = None) -> OCRLine:
    """
    Parse one raw OCR line.
    """
    polygon = raw_line["boundingBox"]
    raw_words = raw_line.get("words", [])
    words = [
        parse_docvqa_ocr_word(word, index=word_idx)
        for word_idx, word in enumerate(raw_words)
    ]

    return OCRLine(
        text=raw_line["text"],
        bbox=BBox.from_polygon(polygon),
        words=words,
        index=index,
        raw_polygon=polygon,
    )


def parse_docvqa_ocr_page(raw_ocr: dict[str, Any]) -> OCRPage:
    """
    Parse a raw Microsoft OCR JSON into OCRPage.

    Raw OCR example:
    {
        "status": "Succeeded",
        "recognitionResults": [
            {
                "page": 1,
                "clockwiseOrientation": 359.96,
                "width": 1692,
                "height": 2245,
                "unit": "pixel",
                "lines": [...]
            }
        ]
    }

    For the current project version, we only parse the first page because
    SP-DocVQA is a single-page task.
    """
    recognition_results = raw_ocr.get("recognitionResults", [])
    if not recognition_results:
        raise ValueError("OCR JSON does not contain recognitionResults")

    first_page = recognition_results[0]
    raw_lines = first_page.get("lines", [])
    lines = [
        parse_docvqa_ocr_line(line, index=line_idx)
        for line_idx, line in enumerate(raw_lines)
    ]

    return OCRPage(
        page_no=_to_int(first_page.get("page")),
        width=_to_int(first_page.get("width")),
        height=_to_int(first_page.get("height")),
        unit=first_page.get("unit"),
        orientation=float(first_page["clockwiseOrientation"])
        if first_page.get("clockwiseOrientation") is not None
        else None,
        status=raw_ocr.get("status"),
        lines=lines,
        source="microsoft_ocr",
    )