# EMR Extractor

从 PDF、JPG 或 PNG 病例中提取 Markdown 与图片，将文本整理为 JSON，并按字段或坐标遮盖敏感信息。

[在线文档](https://www.bj-yan.top/emr-extractor/) · [部署指南](https://www.bj-yan.top/emr-extractor/deployment/)

> [!WARNING]
> 本项目处于 Alpha 阶段，不保证自动识别出全部个人信息，也不等同于完成匿名化。对外分享前必须人工复核输出文件；不要把真实患者资料放进 Issue、测试夹具或 Git 仓库。

## 功能

- 可选择 PaddleOCR-VL 或 MinerU，从 PDF/JPG/PNG 提取 Markdown 和内嵌图片。
- 通过 OpenAI 兼容接口把 Markdown 整理为结构化 JSON。
- 递归遮盖 JSON 中指定名称的字段。
- 按归一化坐标不可逆地遮盖 PDF/JPG/PNG 区域。
- 可选：用 OCR 查找带标签的字段并遮盖其值（版式相关，必须人工复核）。

## 安装

需要 Python 3.10 或更高版本，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/yyyanbj/emr-extractor.git
cd emr-extractor

# 区域遮盖和 JSON 字段遮盖
uv sync

# 加上结构化文本能力
uv sync --extra llm

# PaddleOCR-VL 后端
uv sync --extra ocr

# MinerU 后端
uv sync --extra mineru

# 一次安装全部可选能力
uv sync --extra all
```

OCR 还需要推理引擎。CPU 环境可按 PaddleOCR 官方说明安装 PaddlePaddle：

```bash
uv pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

GPU、其他硬件和新版引擎请以 [PaddleOCR 安装文档](https://www.paddleocr.ai/main/en/version3.x/installation.html) 为准。

## 快速开始

安装后统一使用 `emr-extractor`。查看全部命令：

```bash
uv run emr-extractor --help
```

### 1. 提取 Markdown 和图片

输入可以是单个 PDF/JPG/PNG，也可以是目录。默认使用 PaddleOCR-VL：

```bash
uv run emr-extractor extract records/ artifacts/ \
  --backend paddleocr \
  --recursive
```

需要复杂版面、表格或公式时，也可以切换到 MinerU：

```bash
uv run emr-extractor extract records/ artifacts/ \
  --backend mineru \
  --mineru-tier standard \
  --recursive
```

每个病例写入独立目录，例如 `artifacts/case-001/case-001.md`；所选后端提取的图片和中间产物会保存在同一目录下。

### 2. 整理为 JSON

默认连接本机的 OpenAI 兼容服务 `http://127.0.0.1:11434/v1/`，默认模型为 `qwen3:8b`：

```bash
uv run emr-extractor structure artifacts/ structured/ \
  --recursive \
  --model qwen3:8b
```

也可以指定其他兼容服务：

```bash
OPENAI_API_KEY=your-key uv run emr-extractor structure case.md case.json \
  --base-url https://example.com/v1/ \
  --model your-model
```

远程接口会接收到完整病例文本。只有在数据授权、传输和保存方式满足你的隐私与合规要求时才应使用。

### 3. 遮盖结构化字段

字段名不区分大小写，并会递归匹配嵌套对象和数组。默认遮盖 `Name`、`Phone`、`Address`、`Doctor`：

```bash
uv run emr-extractor redact-fields structured/ safe-json/ \
  --recursive
```

需要其他字段时可覆盖默认列表：

```bash
uv run emr-extractor redact-fields structured/ safe-json/ \
  --fields Name Phone Address Doctor 姓名 电话 地址 医生 "File No."
```

匹配字段的值默认替换为 `[REDACTED]`，键名和其他结构保持不变。

### 4. 遮盖自定义区域

坐标格式为 `[x1, y1, x2, y2]`，取值按页面宽高归一化到 `0..1`。列表会应用到所有 PDF 页面或整张图片：

```bash
uv run emr-extractor redact-areas records/ safe-documents/ \
  --recursive \
  --areas '[[0.65, 0.00, 1.00, 0.20]]'
```

PDF 可以按页配置；页码从 1 开始，支持 `all`、单页、范围和逗号列表：

```bash
uv run emr-extractor redact-areas case.pdf case_redacted.pdf \
  --areas '{"all":[[0.65,0,1,0.12]],"1":[[0,0,0.4,0.2]],"3-5":[[0.1,0.8,0.9,1]]}'
```

PDF 会先栅格化再输出，因此原文字层、批注和元数据不会保留。JPG/PNG 也会重新编码。命令拒绝直接覆盖源文件。

### 5. OCR 自动字段遮盖（可选）

该命令用于标签和值相邻的表单，并依赖本地 OCR 权重：

```bash
uv run emr-extractor download-models \
  --backend paddleocr \
  --model-home models

uv run emr-extractor redact-detected-fields \
  --input-dir records/ \
  --output-dir safe-documents/ \
  --model-home models \
  --recursive
```

该命令也默认查找 `Name`、`Phone`、`Address`、`Doctor` 标签。未检测到目标字段的文件不会生成“安全副本”，而会记录到 `unredacted_files.txt`。不要把“没有检测到”理解为“不含敏感信息”。

## 部署与离线运行

模型可以提前下载到可搬运的统一目录：

```bash
# 二选一，也可使用 --backend all
uv run emr-extractor download-models --backend paddleocr --model-home models
uv run emr-extractor download-models --backend mineru --model-home models

# 断网机器运行
uv run emr-extractor extract records/ artifacts/ \
  --backend mineru \
  --model-home models \
  --offline
```

需要连同 Python wheels 一起搬运时，可在与目标机器相同的操作系统、CPU 架构和 Python 版本上生成完整离线包：

```bash
scripts/build_offline_bundle.sh paddleocr
# 或
scripts/build_offline_bundle.sh mineru
```

在线部署、模型源选择、CPU/GPU 组合、离线包安装和验收流程见 [部署指南](https://github.com/yyyanbj/emr-extractor/blob/main/docs/deployment.md)。

## Python API

```python
from pathlib import Path

from emr_extractor.redaction import parse_areas, redact_documents, redact_fields

safe_data, count = redact_fields(
    {"name": "Alice", "visit": {"phone": "123"}},
    ["name", "phone"],
)

redact_documents(
    Path("case.jpg"),
    Path("case_redacted.jpg"),
    parse_areas([[0.0, 0.0, 1.0, 0.1]]),
)
```

## 项目结构

```text
src/emr_extractor/   可安装的 Python 包与命令行入口
tests/               不含真实患者信息的自动化测试
docs/                部署与离线运行说明
scripts/             离线部署包构建和安装脚本
pyproject.toml       构建、依赖和开发工具配置
CONTRIBUTING.md      贡献指南
SECURITY.md          安全问题报告方式
```

## 开发

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python -m build
```

提交前请确认 `git status` 中没有病例、OCR 输出、模型权重、患者标识或机构内部导出文件。

## 许可证

[Apache License 2.0](LICENSE)
