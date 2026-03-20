"""
omain/schemas.py - 数据模型层（核心数据结构）
这个文件定义了项目的统一数据规范，使用 Pydantic 的 dataclass 来构建高效的数据结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["docvqa", "chartqa", "unknown"]  # DatasetName 取值只能在这里面取，拼错静态检查时就能发现
SplitName = Literal["train", "val", "test", "unknown"]


# BBox - 边界框标准化转换器
@dataclass(slots=True)  # __slots__优化：减少内存占用约 40-50%，提升属性访问速度
class BBox:
    """
    Internal axis-aligned bounding box.

    Note:
    Raw OCR gives an 8-number polygon, for example:
    [x1, y1, x2, y2, x3, y3, x4, y4]

    Inside the project, we normalize it into:
    x0, y0, x1, y1
    """

    x0: float  # 左上角 x 坐标（最小值）
    y0: float  # 左上角 y 坐标（最小值）
    x1: float  # 右下角 x 坐标（最大值）
    y1: float  # 右下角 y 坐标（最大值）

    def __post_init__(self) -> None:  # 钩子函数，实例化类时，会自动调用该方法
        # 数据验证：确保坐标的合法性（x1>x0, y1>y0）
        if self.x1 < self.x0:
            raise ValueError(f"Invalid bbox: x1({self.x1}) < x0({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"Invalid bbox: y1({self.y1}) < y0({self.y0})")

    @property
    def width(self) -> float:
        return self.x1 - self.x0  # 计算宽度：右边界 - 左边界

    @property
    def height(self) -> float:
        return self.y1 - self.y0  # 计算高度：下边界 - 上边界

    @property
    def area(self) -> float:
        return self.width * self.height  # 计算面积：宽 × 高

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)  # 导出为元组格式


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
    def from_polygon(cls, polygon: list[float]) -> "BBox":  # 类方法常常被用作替代构造函数，可以用来创建类的实例。
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


# OCRWord → OCRLine → OCRPage - OCR 结果层级结构
# 三层结构的包含关系：OCRPage 包含多个 OCRLine，每个 OCRLine 包含多个 OCRWord
# 这种设计符合 OCR 的自然层级：页面 → 行 → 词
@dataclass(slots=True)
class OCRWord:
    """One OCR word/token."""

    text: str  # 识别出的文本内容
    bbox: BBox  # 该词的边界框（嵌套使用上面的 BBox 类）
    confidence: str | None = None  # 置信度（预留字段）
    index: int | None = None  # 在行中的索引位置
    raw_polygon: list[float] | None = None  # 保存原始 8 点多边形（用于调试或可视化）

    def __post_init__(self) -> None:
        self.text = self.text.strip()  # 数据清洗：去除首尾空白

    @property
    def is_empty(self) -> bool:
        return not self.text  # 快速判断是否为空词


@dataclass(slots=True)
class OCRLine:
    """One OCR text line."""

    text: str  # 整行的文本内容
    bbox: BBox  # 该行的边界框
    words: list[OCRWord] = field(default_factory=list)  # 包含的词列表（组合关系）
    index: int | None = None  # 在页面中的行号
    raw_polygon: list[float] | None = None

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def n_words(self) -> int:
        return len(self.words)  # 统计词数

    def reconstructed_text(self) -> str:
        """
        从 words 重新构建文本（更可靠）
        原理：当原始 text 可能有误时，可以通过拼接 words.text 来获得准确文本
        """
        if self.words:
            return " ".join(word.text for word in self.words if word.text).strip()
        return self.text  # 如果没有 words，回退到原始 text


@dataclass(slots=True)
class OCRPage:
    """OCR result for one page."""

    page_no: int | None = None  # 页码
    width: int | None = None  # 页面宽度（像素）
    height: int | None = None  # 页面高度（像素）
    unit: str | None = None  # 单位（如"pixel"）
    orientation: float | None = None  # 页面旋转角度
    status: str | None = None  # OCR 识别状态
    lines: list[OCRLine] = field(default_factory=list)  # 包含的所有行
    source: str | None = None  # OCR 引擎来源（如"microsoft_ocr"）

    @property
    def n_lines(self) -> int:
        return len(self.lines)  # 统计行数

    @property
    def n_words(self) -> int:
        return sum(line.n_words for line in self.lines)  # 统计总词数（聚合计算）

    def full_text(self) -> str:
        """
        提取完整文本
        数据流：遍历所有 lines → 调用每行的 reconstructed_text() → 用换行符连接
        """
        return "\n".join(
            line.reconstructed_text()
            for line in self.lines
            if line.reconstructed_text()
        ).strip()


# DocumentSample - 统一的样本表示
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

# DatasetManifest & PredictionRecord - 数据集和预测记录
@dataclass(slots=True)
class DatasetManifest:
    """Dataset-level container."""

    dataset_name: DatasetName
    version: str
    split: SplitName
    samples: list[DocumentSample] = field(default_factory=list)  # 包含的所有样本
    source_files: dict[str, str] = field(default_factory=dict)  # 源文件映射

    def __len__(self) -> int:
        return len(self.samples)  # 支持 len(manifest) 语法

    def add(self, sample: DocumentSample) -> None:
        self.samples.append(sample)  # 添加单个样本


@dataclass(slots=True)
class PredictionRecord:
    """One prediction output."""

    question_id: str
    prediction: str
    dataset_name: DatasetName = "unknown"
    # meta是一个字段，类型是dict[str, Any]。
    # 使用field来配置这个字段。
    # default_factory = dict 表示当创建实例时，如果没有显式传入meta的值，就会调用dict()来生成一个新的空字典作为默认值。
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.question_id = str(self.question_id).strip()
        self.prediction = self.prediction.strip()
        if not self.question_id:
            raise ValueError("question_id must not be empty")