# 使用指南

## 安装

需要 Python 3.10 或更高版本，推荐 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/yyyanbj/emr-extractor.git
cd emr-extractor

# 基础的 JSON 字段与文档区域遮盖
uv sync

# 按需安装
uv sync --extra llm
uv sync --extra ocr
uv sync --extra mineru
```

PaddleOCR 还需要与硬件匹配的 PaddlePaddle 推理引擎；完整安装方式见[部署指南](deployment.md)。

## 提取 Markdown 与图片

PaddleOCR-VL 是默认后端：

```bash
uv run emr-extractor extract records/ artifacts/ \
  --backend paddleocr \
  --recursive
```

复杂版面、表格或公式可以选择 MinerU：

```bash
uv run emr-extractor extract records/ artifacts/ \
  --backend mineru \
  --mineru-tier standard \
  --recursive
```

MinerU 提供四个质量档：`flash`、`basic`、`standard`、`advanced`。CPU 环境优先尝试 `basic`；复杂布局使用默认的 `standard`。

## 结构化为 JSON

默认连接本机的 OpenAI 兼容服务 `http://127.0.0.1:11434/v1/`：

```bash
uv run emr-extractor structure artifacts/ structured/ \
  --recursive \
  --model qwen3:8b
```

使用远程接口时，完整病例文本会离开本机。只有满足数据授权、传输和保存要求时才能指定远程 `--base-url`。

## 字段脱敏

不传 `--fields` 时，默认遮盖 `Name`、`Phone`、`Address`、`Doctor`：

```bash
uv run emr-extractor redact-fields structured/ safe-json/ --recursive
```

也可以覆盖默认字段：

```bash
uv run emr-extractor redact-fields structured/ safe-json/ \
  --fields Name Phone Address Doctor 姓名 电话 地址 医生 "File No."
```

字段名不区分大小写，并会递归匹配嵌套对象和数组。

## 区域遮盖

坐标格式为 `[x1, y1, x2, y2]`，按页面宽高归一化到 `0..1`：

```bash
uv run emr-extractor redact-areas records/ safe-documents/ \
  --recursive \
  --areas '[[0.65,0,1,0.2]]'
```

PDF 可以按页配置，页码从 1 开始：

```bash
uv run emr-extractor redact-areas case.pdf case_redacted.pdf \
  --areas '{"all":[[0.65,0,1,0.12]],"1":[[0,0,0.4,0.2]],"3-5":[[0.1,0.8,0.9,1]]}'
```

PDF 会被栅格化，JPG/PNG 会重新编码；命令拒绝直接覆盖源文件。

## OCR 标签字段遮盖

该命令默认查找 `Name`、`Phone`、`Address`、`Doctor` 标签：

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

未检测到字段的文件不会生成安全副本，并会记录到 `unredacted_files.txt`。没有检测到不代表文件不含敏感信息。

## 查看命令帮助

```bash
uv run emr-extractor --help
uv run emr-extractor extract --help
uv run emr-extractor download-models --help
```
