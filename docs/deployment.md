# 部署指南

EMR Extractor 的文档解析始终在本机执行，用户可在 PaddleOCR-VL 和 MinerU 之间选择。结构化步骤是否调用远程 LLM，则由 `structure --base-url` 决定。

## 后端选择

| 后端 | 适合场景 | 安装 | 默认模型准备 |
| --- | --- | --- | --- |
| PaddleOCR-VL | 通用 PDF/图片转 Markdown，与现有流程兼容 | `uv sync --frozen --extra ocr` | `download-models --backend paddleocr` |
| MinerU | 复杂版面、表格、公式；需要分档控制质量和资源 | `uv sync --frozen --extra mineru` | `download-models --backend mineru` |

两个后端都支持 PDF、JPG、JPEG 和 PNG。MinerU 的 `basic` 档只需要小模型，适合 CPU；`standard` 还需要 VLM，质量更高。默认使用 `standard + ONNX + llama.cpp`。

## 在线部署

### PaddleOCR-VL

```bash
uv sync --frozen --extra llm --extra ocr

# CPU 推理引擎；GPU 请根据驱动选择 PaddlePaddle 官方 wheel
uv pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

uv run emr-extractor download-models \
  --backend paddleocr \
  --model-home /srv/emr-extractor/models
```

无法访问 Hugging Face 时可加 `--paddle-source bos`。运行：

```bash
uv run emr-extractor extract /srv/input /srv/output \
  --backend paddleocr \
  --model-home /srv/emr-extractor/models \
  --recursive
```

### MinerU

```bash
uv sync --frozen --extra llm --extra mineru

uv run emr-extractor download-models \
  --backend mineru \
  --model-home /srv/emr-extractor/models \
  --mineru-tier standard \
  --mineru-small-backend onnx \
  --mineru-vlm-engine llama-cpp
```

国内网络可以加 `--mineru-source modelscope`。运行：

```bash
uv run emr-extractor extract /srv/input /srv/output \
  --backend mineru \
  --mineru-tier standard \
  --model-home /srv/emr-extractor/models \
  --recursive
```

NVIDIA 高吞吐部署可安装 `mineru[full]`，并在下载和运行环境中保持同一组 `small_backend` 与 `vlm_engine`。具体 CUDA、vLLM 或 LMDeploy 组合以 [MinerU 扩展模块文档](https://opendatalab.github.io/MinerU/quick_start/extension_modules/) 为准。

## 离线部署

### 1. 在联网构建机准备

构建机必须与离线目标机使用相同的操作系统、CPU 架构和 Python 3.12。脚本会构建项目 wheel，下载锁定的依赖 wheel，下载并验证所选后端模型：

```bash
scripts/build_offline_bundle.sh paddleocr /tmp/emr-paddle-offline
# 或
scripts/build_offline_bundle.sh mineru /tmp/emr-mineru-offline
```

PaddleOCR 脚本准备 CPU 版 PaddlePaddle；GPU wheel 与主机驱动相关，应按官方安装页替换。MinerU 脚本准备 `standard + ONNX + llama.cpp`，避免依赖目标机 GPU。

生成目录包含：

```text
BACKEND
requirements.txt
wheels/
models/
install_offline.sh
deployment.md
```

把整个目录复制到离线机器，不要只复制模型目录。

### 2. 在离线目标机安装

```bash
cd /path/to/emr-paddle-offline
bash install_offline.sh
```

脚本只使用 `wheels/`，不会访问包索引，并在安装后执行依赖一致性检查。

### 3. 强制离线运行

```bash
.venv/bin/emr-extractor extract /srv/input /srv/output \
  --backend paddleocr \
  --model-home "$PWD/models" \
  --offline \
  --recursive
```

MinerU 将 `--backend` 改为 `mineru`。`--offline` 会令 MinerU 只接受本地模型，并启用 PaddleOCR/Hugging Face 的离线缓存模式；模型缺失时直接失败，不会临时联网下载。

Windows 没有 Bash 时，按 `install_offline.sh` 中相同步骤执行：创建 Python 3.12 虚拟环境，使用 `pip install --no-index --find-links wheels -r requirements.txt` 安装，再从 `wheels/` 安装 `emr_extractor-*.whl`。

## 验收

使用完全虚构且不含患者信息的小样本进行端到端验收：

```bash
.venv/bin/emr-extractor --version
.venv/bin/emr-extractor extract synthetic.pdf smoke-output \
  --backend mineru \
  --model-home "$PWD/models" \
  --offline
.venv/bin/emr-extractor redact-fields synthetic.json safe.json
```

检查 Markdown、图片、脱敏 JSON 和最终遮盖文档。自动字段检测可能漏检，生产输出仍必须人工复核。

## 权限与数据目录

- 输入目录只读挂载；输出目录和 `model-home` 分开授权。
- 不要把病例、日志、模型或输出放入 Git 仓库。
- 远程 LLM 会收到完整 Markdown；未完成隐私评估前只使用本机兼容服务。
- MinerU 采用独立的开源许可证，部署前应阅读其[许可证和使用条件](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)。
