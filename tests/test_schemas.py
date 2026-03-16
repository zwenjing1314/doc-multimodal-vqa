import json
from pathlib import Path

from mm_docvqa.domain.schemas import (
    BBox,
    DocumentSample,
    OCRLine,
    OCRPage,
    OCRWord,
    PredictionRecord,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_bbox_area() -> None:
    box = BBox(0, 0, 10, 5)
    assert box.width == 10
    assert box.height == 5
    assert box.area == 50


def test_bbox_from_polygon() -> None:
    polygon = [622, 138, 1137, 136, 1138, 167, 622, 168]
    box = BBox.from_polygon(polygon)
    assert box.x0 == 622
    assert box.y0 == 136
    assert box.x1 == 1138
    assert box.y1 == 168


def test_ocr_line_reconstructed_text() -> None:
    line = OCRLine(
        text="R. J. REYNOLDS TOBACCO COMPANY",
        bbox=BBox(622, 136, 1138, 168),
        words=[
            OCRWord(text="R.", bbox=BBox(622, 138, 656, 168)),
            OCRWord(text="J.", bbox=BBox(661, 139, 687, 168)),
            OCRWord(text="REYNOLDS", bbox=BBox(691, 140, 849, 168)),
            OCRWord(text="TOBACCO", bbox=BBox(855, 140, 991, 168)),
            OCRWord(text="COMPANY", bbox=BBox(998, 137, 1138, 167)),
        ],
    )
    assert "REYNOLDS" in line.reconstructed_text()


def test_ocr_page_full_text() -> None:
    line1 = OCRLine(
        text="R. J. REYNOLDS TOBACCO COMPANY",
        bbox=BBox(622, 136, 1138, 168),
        words=[
            OCRWord(text="R.", bbox=BBox(622, 138, 656, 168)),
            OCRWord(text="J.", bbox=BBox(661, 139, 687, 168)),
        ],
    )
    line2 = OCRLine(
        text="Request For Taxpayer Identification Number and Certification",
        bbox=BBox(143, 254, 797, 282),
        words=[
            OCRWord(text="Request", bbox=BBox(144, 256, 234, 282)),
            OCRWord(text="For", bbox=BBox(239, 256, 278, 282)),
        ],
    )

    page = OCRPage(
        page_no=1,
        width=1692,
        height=2245,
        unit="pixel",
        orientation=359.96,
        status="Succeeded",
        source="microsoft_ocr",
        lines=[line1, line2],
    )

    text = page.full_text()
    assert "R." in text
    assert "Request" in text


def test_document_sample_basic_behavior() -> None:
    sample = DocumentSample(
        dataset_name="docvqa",
        split="train",
        question_id="337",
        question=" what is the date mentioned in this letter? ",
        image_path="documents/xnbl0037_1.png",
        answers=[" 1/8/93 ", "1/8/93"],
        question_types=["handwritten", "form"],
        doc_numeric_id=279,
        doc_id="xnbl0037",
        page_no=1,
    )

    assert sample.primary_answer == "1/8/93"
    assert "1/8/93" in sample.normalized_answers
    assert sample.image_name == "xnbl0037_1.png"
    assert sample.doc_numeric_id == 279
    assert sample.doc_id == "xnbl0037"
    assert sample.page_no == 1
    assert sample.question_types == ["handwritten", "form"]


def test_prediction_record() -> None:
    record = PredictionRecord(
        question_id="337",
        prediction="1/8/93",
        dataset_name="docvqa",
    )
    assert record.question_id == "337"
    assert record.prediction == "1/8/93"


def test_fake_docvqa_fixture_can_be_loaded() -> None:
    data = json.loads((FIXTURE_DIR / "fake_docvqa_sample.json").read_text(encoding="utf-8"))
    assert data["questionId"] == 337
    assert data["image"] == "documents/xnbl0037_1.png"


def test_fake_docvqa_ocr_fixture_can_be_loaded() -> None:
    data = json.loads((FIXTURE_DIR / "fake_docvqa_ocr.json").read_text(encoding="utf-8"))
    assert data["status"] == "Succeeded"
    assert "recognitionResults" in data
    assert len(data["recognitionResults"]) == 1
    assert len(data["recognitionResults"][0]["lines"]) >= 1