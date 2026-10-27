# Contributing

感谢你改进 EMR Extractor。

## 开发流程

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python -m build
```

请保持改动聚焦，并为新的分支、循环、解析或隐私相关逻辑补充小型测试。公共 API、命令参数或输出格式发生变化时，同时更新 README。

## 数据安全

- 不要提交真实或可重新识别的患者数据，包括文件名、截图、日志和 OCR 输出。
- 测试数据必须完全虚构，且不能由真实病例简单替换姓名得到。
- Issue 和 Pull Request 中也不要粘贴敏感文本；必要时用最小的合成示例复现问题。
- 新的遮盖逻辑必须在“未匹配”时给出明确结果，不能暗示文件已经安全匿名化。

## Pull Request

说明问题、最小解决方案和验证方式。涉及遮盖行为时，请同时说明漏遮盖与误遮盖的风险。
