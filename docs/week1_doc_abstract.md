# src/mm_docvqa/domain/schemas.py（数据模型层）
定义了一组“统一数据规范”的 dataclass（没有显式的接口/抽象类）：
```text
DatasetName / SplitName：typing.Literal 类型枚举（用于静态检查 dataset/split 名称）
class BBox
    属性：x0,y0,x1,y1
    校验：__post_init__ 确保 x1>=x0、y1>=y0
    方法/属性：width/height/area、as_tuple()、@classmethod from_polygon(polygon8)
class OCRWord
    属性：text、bbox: BBox、confidence、index、raw_polygon
    __post_init__：对 text.strip()
    属性：is_empty
class OCRLine
    属性：text、bbox、words: list[OCRWord]、index、raw_polygon
    __post_init__：text.strip()
    属性：n_words
    方法：reconstructed_text()（优先用 words.text 重建，空则回退 self.text）
class OCRPage
    属性：page_no,width,height,unit,orientation,status,lines,source
    属性：n_lines、n_words
    方法：full_text()（把每行的 reconstructed_text() 用换行拼起来）
class DocumentSample（统一样本）
    关键字段：dataset_name, split, question_id, question, image_path, answers, question_types
OCR/文档扩展字段：doc_numeric_id、doc_id、page_no、ocr_page、meta: dict
    __post_init__：清理字符串、清理答案/类型、并校验 question_id/question/image_path 非空
    属性：has_answers、has_ocr、primary_answer、normalized_answers、image_name
class DatasetManifest（数据集清单）
    字段：dataset_name, version, split, samples, source_files
    方法：__len__()、add(sample)
class PredictionRecord（预测输出记录）
    字段：question_id, prediction, dataset_name="unknown", meta
    __post_init__：清理并校验 question_id 非空
```

# src/mm_docvqa/data/parser_docvqa.py（解析层：原始 JSON -> 规范结构）
包含一组“把原始字段映射成上面 schemas 的函数”，核心函数：
```text
_to_int(value) -> int | None：安全转 int（None/空字符串 -> None）
parse_docvqa_sample(raw_sample, default_split=None) -> DocumentSample
    强制：dataset_name="docvqa"
    split 取：raw_sample["data_split"] or default_split or "unknown"
    字段映射：例如 questionId -> question_id、image -> image_path、docId -> doc_numeric_id 等
    meta：保留原始 data_split 到 meta["raw_data_split"]
parse_docvqa_manifest(raw_manifest) -> DatasetManifest
    从 raw_manifest["dataset_split"] 得到 split
    遍历 raw_manifest["data"]，逐条 parse_docvqa_sample
    生成 DatasetManifest(dataset_name="docvqa", version=..., split=..., samples=...)
OCR 三级解析（Word -> Line -> Page）
    parse_docvqa_ocr_word(raw_word, index=None) -> OCRWord
        boundingBox(8点多边形) -> BBox.from_polygon()
    parse_docvqa_ocr_line(raw_line, index=None) -> OCRLine
        words[] -> list[OCRWord]
    parse_docvqa_ocr_page(raw_ocr) -> OCRPage
        从 raw_ocr["recognitionResults"][0] 只取第一页（注释里说明是单页任务）
        lines[] -> list[OCRLine]
        若 recognitionResults 为空会 raise ValueError
```

# src/mm_docvqa/data/loader.py（加载层：文件路径 & 串联解析）
提供“高层 API”，把 JSON 文件加载出来并拼到统一结构中：
```text
@dataclass(frozen=True) class DocVQAPaths
    from_root(root) 推导目录结构：
        qas_dir = root/"spdocvqa_qas"
        images_dir = root/"spdocvqa_images"
        ocr_dir = root/"spdocvqa_ocr"
load_json(path) -> dict：json.load 一次性读入内存
load_docvqa_manifest(qas_json_path) -> DatasetManifest
    load_json + parse_docvqa_manifest
资源路径解析与“附加”：
    resolve_docvqa_image_path(images_dir, relative_image_path)
    resolve_docvqa_ocr_path(ocr_dir, relative_image_path)
        通过图片 stem 推导 OCR 文件：image_stem + ".json"
    attach_absolute_image_paths(manifest, images_dir)：原地把 sample.image_path 改成绝对路径
    attach_ocr_paths(manifest, ocr_dir)：把 sample.meta["ocr_path"] 填好
load_docvqa_manifest_with_assets(...)
    串联：load_docvqa_manifest -> attach_absolute_image_paths -> attach_ocr_paths
load_docvqa_ocr_page(ocr_json_path) -> OCRPage
    load_json + parse_docvqa_ocr_page
get_default_docvqa_qas_file(qas_dir, split) -> Path
    split->文件名映射：
        train: train_v1.0_withQT.json
        val: val_v1.0_withQT.json
        test: test_v1.0.json
```

# scripts/inspect_docvqa_train.py（脚本入口/调用链演示）
它直接演示如何从 datasets/docvqa/... 读入并解析。 
调用流程（高层）：
```text
1. docvqa_root = project_root / "datasets" / "docvqa"
2. paths = DocVQAPaths.from_root(docvqa_root)
3. train_qas_file = get_default_docvqa_qas_file(paths.qas_dir, "train")
4. manifest = load_docvqa_manifest_with_assets(...)
5. 打印 manifest.samples[i] 的字段
6. 取 manifest.samples[0].meta["ocr_path"]
7. ocr_page = load_docvqa_ocr_page(first_ocr_path)
8. 打印 ocr_page 汇总 + 前几行 line.text
```
# 测试（理解“期望输入长什么样”）
tests/test_parser.py：验证 parse_docvqa_sample/manifest/ocr_page 的字段映射结果
tests/test_schemas.py：验证 bbox、OCR 结构重建文本等方法的行为
tests/fixtures/*.json：小型的样例输入（结构与大 JSON 一致，只是很小）

# 这个文件的关系
```text
┌─────────────────────────────────────────────────────────────┐
│                    原始数据层 (Raw Data)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  QAS JSON    │  │  OCR JSON    │  │   Images     │       │
│  │  (标注文件)   │  │  (识别结果)   │  │  (图片文件)   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘       │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   解析层 (parser_docvqa.py)                  │
│  作用：将原始 JSON → 标准 Python 对象                          │
│  ┌────────────────────────────────────────────────┐         │
│  │ parse_docvqa_manifest()                        │         │
│  │   ↓ 调用                                       │         │
│  │ parse_docvqa_sample() → DocumentSample        │         │
│  │                                                │         │
│  │ parse_docvqa_ocr_page()                        │         │
│  │   ↓ 递归调用                                   │         │
│  │ parse_docvqa_ocr_line() → OCRLine             │         │
│  │     ↓ 递归调用                                 │         │
│  │   parse_docvqa_ocr_word() → OCRWord           │         │
│  └────────────────────────────────────────────────┘         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  数据模型层 (domain/schemas.py)               │
│  作用：定义标准数据结构（容器）                                │
│  ┌────────────────────────────────────────────────┐         │
│  │ DatasetManifest ← 装所有样本的"大箱子"          │         │
│  │   └─ samples: list[DocumentSample]             │         │
│  │        └─ DocumentSample ← 单个 QA 样本          │         │
│  │            ├─ question, answers, image_path    │         │
│  │            └─ ocr_page: OCRPage ← OCR 结果      │         │
│  │                └─ lines: list[OCRLine]         │         │
│  │                    └─ words: list[OCRWord]     │         │
│  └────────────────────────────────────────────────┘         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 加载层 (loader.py)                           │
│  作用：提供高级 API，协调 parser 和 schemas                    │
│  ┌────────────────────────────────────────────────┐         │
│  │ load_docvqa_manifest()                         │         │
│  │   → 调用 parser，返回 DatasetManifest           │         │ 
│  │                                                │         │
│  │ load_docvqa_manifest_with_assets()             │         │
│  │   → 加载 manifest + OCR + 图片路径               │         │
│  │                                                │         │
│  │ load_docvqa_ocr_page()                         │         │
│  │   → 调用 parser，返回 OCRPage                   │         │
│  └────────────────────────────────────────────────┘         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              应用层 (scripts/inspect_docvqa_train.py)        │
│  作用：调用 loader 的 API，显示/使用数据                       │
│  ┌────────────────────────────────────────────────┐         │
│  │ main()                                         │         │
│  │   → 调用 loader 函数                            │         │
│  │   → 访问 manifest 的属性                        │         │
│  │   → print() 输出                               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘

```