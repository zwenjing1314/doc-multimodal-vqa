from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["docvqa", "chartqa", "unknown"]  # DatasetName 取值只能在这里面取，拼错静态检查时就能发现
SplitName = Literal["train", "val", "test", "unknown"]


@dataclass(slots=True)
class BBox:
    """
    Internal axis-aligned bounding box.

    Note:
    Raw OCR gives an 8-number polygon, for example:
    [x1, y1, x2, y2, x3, y3, x4, y4]

    Inside the project, we normalize it into:
    x0, y0, x1, y1
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0:
            raise ValueError(f"Invalid bbox: x1({self.x1}) < x0({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"Invalid bbox: y1({self.y1}) < y0({self.y0})")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    """
    @classmethod 内存中的结构：
    BBox 类对象（存在于内存中）
    ├── __init__ (由 dataclass 自动生成)
    ├── from_polygon (类方法)
    ├── height (property)
    └── width (property)
    ...
    """
    @classmethod
    def from_polygon(cls, polygon: list[float]) -> "BBox":
        """
        Convert an OCR polygon with 8 numbers into an axis-aligned bbox.

        Example polygon:
        [622, 138, 1137, 136, 1138, 167, 622, 168]
        """
        if len(polygon) != 8:
            raise ValueError(f"Polygon must have 8 numbers, got {len(polygon)}")

        xs = polygon[0::2]
        ys = polygon[1::2]
        return cls(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
        )


@dataclass(slots=True)
class OCRWord:
    """One OCR word/token."""

    text: str
    bbox: BBox
    confidence: str | None = None
    index: int | None = None
    raw_polygon: list[float] | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def is_empty(self) -> bool:
        return not self.text


@dataclass(slots=True)
class OCRLine:
    """One OCR text line."""

    text: str
    bbox: BBox
    words: list[OCRWord] = field(default_factory=list)
    index: int | None = None
    raw_polygon: list[float] | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def n_words(self) -> int:
        return len(self.words)

    def reconstructed_text(self) -> str:
        if self.words:
            return " ".join(word.text for word in self.words if word.text).strip()
        return self.text


@dataclass(slots=True)
class OCRPage:
    """OCR result for one page."""

    page_no: int | None = None
    width: int | None = None
    height: int | None = None
    unit: str | None = None
    orientation: float | None = None
    status: str | None = None
    lines: list[OCRLine] = field(default_factory=list)
    source: str | None = None  # e.g. "microsoft_ocr"

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_words(self) -> int:
        return sum(line.n_words for line in self.lines)

    def full_text(self) -> str:
        return "\n".join(
            line.reconstructed_text()
            for line in self.lines
            if line.reconstructed_text()
        ).strip()


@dataclass(slots=True)
class DocumentSample:
    """
    Internal canonical sample.

    This is the unified format used inside the project after parsing raw data.
    It should not copy raw JSON names directly.
    """

    dataset_name: DatasetName
    split: SplitName
    question_id: str
    question: str
    image_path: str
    answers: list[str] = field(default_factory=list)

    question_types: list[str] = field(default_factory=list)

    # From raw DocVQA:
    # docId -> numeric id
    # ucsf_document_id -> document string id
    # ucsf_document_page_no -> page number
    doc_numeric_id: int | None = None
    doc_id: str | None = None
    page_no: int | None = None

    ocr_page: OCRPage | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id).strip()
        self.question = self.question.strip()
        self.image_path = str(self.image_path).strip()

        cleaned_answers: list[str] = []
        for answer in self.answers:
            if isinstance(answer, str):
                answer = answer.strip()
                if answer:
                    cleaned_answers.append(answer)
        self.answers = cleaned_answers

        self.question_types = [q.strip() for q in self.question_types if q.strip()]

        if not self.question_id:
            raise ValueError("question_id must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")
        if not self.image_path:
            raise ValueError("image_path must not be empty")

    @property
    def has_answers(self) -> bool:
        return len(self.answers) > 0

    @property
    def has_ocr(self) -> bool:
        return self.ocr_page is not None

    @property
    def primary_answer(self) -> str | None:
        return self.answers[0] if self.answers else None

    @property
    def normalized_answers(self) -> set[str]:
        return {answer.lower().strip() for answer in self.answers if answer.strip()}

    @property
    def image_name(self) -> str:
        return Path(self.image_path).name


@dataclass(slots=True)
class DatasetManifest:
    """Dataset-level container."""

    dataset_name: DatasetName
    version: str
    split: SplitName
    samples: list[DocumentSample] = field(default_factory=list)
    source_files: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: DocumentSample) -> None:
        self.samples.append(sample)


@dataclass(slots=True)
class PredictionRecord:
    """One prediction output."""

    question_id: str
    prediction: str
    dataset_name: DatasetName = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id).strip()
        self.prediction = self.prediction.strip()
        if not self.question_id:
            raise ValueError("question_id must not be empty")