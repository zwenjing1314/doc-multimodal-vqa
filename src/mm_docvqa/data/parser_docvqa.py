"""
parser_docvqa.py - 数据解析层（原始数据 → 标准格式）
这个文件负责将原始 DocVQA JSON 数据转换为上面定义的统一 Schema。
"""
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

# parse_docvqa_sample - 单样本解析器
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
        转换逻辑：
    1. 提取 dataset_name = "docvqa"（硬编码，因为这是 DocVQA 专用解析器）
    2. 确定 split：优先使用 raw_sample["data_split"]，否则用 default_split
    3. 字段映射：将原始字段名映射到 DocumentSample 的标准字段
    4. 类型转换：questionId 等数字转为字符串，docId 等用_to_int 处理
    5. 保留元数据：将原始 data_split 存入 meta 以备后用
    """
    split = raw_sample.get("data_split") or default_split or "unknown"

    return DocumentSample(
        dataset_name="docvqa",
        split=split,
        question_id=str(raw_sample["questionId"]),  # 转字符串保证一致性
        question=raw_sample["question"],
        image_path=raw_sample["image"],
        answers=raw_sample.get("answers", []),  # 可选字段，默认空列表
        question_types=raw_sample.get("question_types", []),
        doc_numeric_id=_to_int(raw_sample.get("docId")),  # 安全转换
        doc_id=raw_sample.get("ucsf_document_id"),
        page_no=_to_int(raw_sample.get("ucsf_document_page_no")),
        meta={
            "raw_data_split": raw_sample.get("data_split"),  # 保存原始信息
        },
    )


# parse_docvqa_manifest - 清单解析器
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
        处理流程：
    1. 提取 dataset_split（默认为"unknown"）
    2. 遍历 data 数组，对每个样本调用 parse_docvqa_sample
    3. 创建 DatasetManifest，注意 dataset_name 固定为"docvqa"（内部统一名称）
    """
    dataset_split = raw_manifest.get("dataset_split", "unknown")
    samples = [
        parse_docvqa_sample(sample, default_split=dataset_split)
        for sample in raw_manifest.get("data", [])
    ]

    return DatasetManifest(
        dataset_name="docvqa",  # 内部统一名称（不同于原始的"SP-DocVQA"）
        version=str(raw_manifest.get("dataset_version", "unknown")),
        split=dataset_split,
        samples=samples,
        source_files={},  # 留空，后续可由 loader 填充
    )

# OCR 数据解析器（三级解析：Word → Line → Page）
def parse_docvqa_ocr_word(raw_word: dict[str, Any], index: int | None = None) -> OCRWord:
    """
    Parse one raw OCR word.
        处理步骤：
    1. 提取 boundingBox 转为 BBox
    2. 遍历 words 数组，对每个词调用 parse_docvqa_ocr_word（递归分解）
    3. 创建 OCRLine，包含解析后的 words 列表
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
        处理步骤：
    1. 提取 boundingBox 转为 BBox
    2. 遍历 words 数组，对每个词调用 parse_docvqa_ocr_word（递归分解）
    3. 创建 OCRLine，包含解析后的 words 列表
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
        关键设计决策：
    - 只处理第一页（因为 SP-DocVQA 是单页任务）
    - 如果 recognitionResults 为空，抛出异常（数据完整性检查）

    处理流程：
    1. 提取 recognitionResults[0]（第一页）
    2. 遍历 lines 数组，调用 parse_docvqa_ocr_line 解析每行
    3. 提取页面元数据（宽高、方向等）
    4. 创建 OCRPage 对象
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