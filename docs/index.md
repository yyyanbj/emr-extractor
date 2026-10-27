# EMR Extractor

EMR Extractor 用于从 PDF、JPG 或 PNG 病例中提取 Markdown 与图片，将文本整理为结构化 JSON，并按字段或坐标遮盖敏感信息。

!!! warning "隐私提示"

    本项目不保证自动识别出全部个人信息，也不等同于完成匿名化。任何输出在共享或公开前都必须人工复核，不要把真实患者资料提交到 Git、Issue 或测试数据中。

## 主要能力

- **文档提取**：按需选择 PaddleOCR-VL 或 MinerU。
- **文本结构化**：通过 OpenAI 兼容接口将 Markdown 转换为 JSON。
- **字段脱敏**：默认递归遮盖 `Name`、`Phone`、`Address`、`Doctor`。
- **区域遮盖**：按归一化坐标不可逆地处理 PDF/JPG/PNG。
- **离线部署**：提前准备依赖 wheels 与模型，在断网环境强制使用本地资源。

## 从这里开始

1. 阅读[使用指南](usage.md)，完成安装与第一个处理流程。
2. 根据文档特点选择 PaddleOCR-VL 或 MinerU。
3. 生产或隔离网络环境参考[部署与离线运行](deployment.md)。

源代码、问题跟踪与贡献说明位于 [GitHub 仓库](https://github.com/yyyanbj/emr-extractor)。
