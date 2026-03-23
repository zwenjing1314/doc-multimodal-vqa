from __future__ import annotations

from pathlib import Path

from mm_docvqa.data.loader import (
    DocVQAPaths,
    get_default_docvqa_qas_file,
    load_docvqa_manifest_with_assets,
    load_docvqa_ocr_page,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]  # 获取当前文件所在目录，将路径转化为绝对路径并往上跳两级返回祖父目录的绝对路径
    docvqa_root = project_root / "datasets" / "docvqa"  # 获取数据集根目录

    paths = DocVQAPaths.from_root(docvqa_root)  # 创建 DocVQAPaths 对象
    train_qas_file = get_default_docvqa_qas_file(paths.qas_dir, "train")  # 获取 train 对应的训练集标注文件

    print("=" * 80)
    print("DocVQA dataset root:", paths.root)
    print("Train annotation file:", train_qas_file)
    print("=" * 80)

    # 完成数据集级别的容器构建，里面的所有数据都被封装在 DatasetManifest 类中
    manifest = load_docvqa_manifest_with_assets(
        qas_json_path=train_qas_file,
        images_dir=paths.images_dir,
        ocr_dir=paths.ocr_dir,
    )

    print(f"Dataset name: {manifest.dataset_name}")
    print(f"Version: {manifest.version}")
    print(f"Split: {manifest.split}")
    print(f"Number of samples: {len(manifest.samples)}")
    print("=" * 80)

    num_to_show = min(5, len(manifest.samples))
    for idx in range(num_to_show):
        sample = manifest.samples[idx]
        print(f"[Sample {idx}]")
        print("question_id     :", sample.question_id)
        print("question        :", sample.question)
        print("answers         :", sample.answers)
        print("question_types  :", sample.question_types)
        print("doc_numeric_id  :", sample.doc_numeric_id)
        print("doc_id          :", sample.doc_id)
        print("page_no         :", sample.page_no)
        print("image_path      :", sample.image_path)
        print("ocr_path        :", sample.meta.get("ocr_path"))
        print("-" * 80)

    print("\nNow inspect the first sample's OCR file...")
    first_sample = manifest.samples[0]
    first_ocr_path = first_sample.meta["ocr_path"]

    if not Path(first_ocr_path).exists():
        print(f"OCR file not found: {first_ocr_path}")
        return

    ocr_page = load_docvqa_ocr_page(first_ocr_path)

    print("=" * 80)
    print("First OCR page summary")
    print("page_no         :", ocr_page.page_no)
    print("width           :", ocr_page.width)
    print("height          :", ocr_page.height)
    print("unit            :", ocr_page.unit)
    print("orientation     :", ocr_page.orientation)
    print("status          :", ocr_page.status)
    print("source          :", ocr_page.source)
    print("n_lines         :", ocr_page.n_lines)
    print("n_words         :", ocr_page.n_words)
    print("-" * 80)

    preview_lines = ocr_page.lines[:3]
    for i, line in enumerate(preview_lines):
        print(f"OCR line {i}: {line.text}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()