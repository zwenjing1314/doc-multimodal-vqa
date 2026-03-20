# 完成任务
1. 创建 pyproject.toml 文件并添加依赖项
2. 创建 .pre-commit-config.yaml 文件并添加配置项


# 执行命令
完成 pyproject.toml 文件和 .pre-commit-config.yaml 文件的创建后，执行以下命令
```bash
pre-commit install
pre-commit run --all-files
```

# 创建以下文件
```text
1) src/mm_docvqa/domain/schemas.py
2) tests/test_schemas.py
3) tests/fixtures/fake_docvqa_sample.json
4) tests/fixtures/fake_docvqa_ocr.json
```

运行
```bash
pytest tests/test_schemas.py
```

# 创建以下文件
```text
1) src/mm_docvqa/data/parser_docvqa.py
2) src/mm_docvqa/data/loader.py
3) tests/test_parser.py
```

先运行：
```bash
pytest tests/test_parser.py
```

然后再跑:
```bash
pytest
```

# 新增/修改文件
```text
1) 更新 src/mm_docvqa/data/loader.py
2）新增 scripts/inspect_docvqa_train.py
```

在项目根目录下执行以下命令：
```bash
python scripts/inspect_docvqa_train.py
```