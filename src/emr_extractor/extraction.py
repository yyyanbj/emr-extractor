from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from emr_extractor.models import configure_model_environment

SUPPORTED_DOCUMENTS = {".pdf", ".jpg", ".jpeg", ".png"}
EXTRACTION_BACKENDS = ("paddleocr", "mineru")
MINERU_TIERS = ("flash", "basic", "standard", "advanced")
MINERU_OCR_MODES = ("auto", "txt", "ocr")
MINERU_SMALL_BACKENDS = ("auto", "onnx", "torch")
MINERU_VLM_ENGINES = ("auto", "llama-cpp", "vllm", "lmdeploy", "mlx")


def iter_documents(path: Path, recursive: bool = False) -> list[Path]:
    """Return supported documents in a stable order."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_DOCUMENTS:
            raise ValueError(f"unsupported input type: {path.suffix or '<none>'}")
        return [path]

    candidates: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        item for item in candidates if item.is_file() and item.suffix.lower() in SUPPORTED_DOCUMENTS
    )


class DocumentExtractor:
    """Extract Markdown and embedded images with a selected local backend."""

    def __init__(
        self,
        pipeline: Any | None = None,
        *,
        backend: str = "paddleocr",
        model_home: Path | None = None,
        offline: bool = False,
        mineru_tier: str = "standard",
        mineru_ocr_mode: str = "auto",
        mineru_small_backend: str | None = None,
        mineru_vlm_engine: str | None = None,
        mineru_parser: Any | None = None,
        mineru_writer: Any | None = None,
    ) -> None:
        if backend not in EXTRACTION_BACKENDS:
            raise ValueError(f"unsupported extraction backend: {backend}")
        if mineru_tier not in MINERU_TIERS:
            raise ValueError(f"unsupported MinerU tier: {mineru_tier}")
        if mineru_ocr_mode not in MINERU_OCR_MODES:
            raise ValueError(f"unsupported MinerU OCR mode: {mineru_ocr_mode}")
        if mineru_small_backend is not None and mineru_small_backend not in MINERU_SMALL_BACKENDS:
            raise ValueError(f"unsupported MinerU small-model backend: {mineru_small_backend}")
        if mineru_vlm_engine is not None and mineru_vlm_engine not in MINERU_VLM_ENGINES:
            raise ValueError(f"unsupported MinerU VLM engine: {mineru_vlm_engine}")
        self.backend = backend
        self._pipeline = pipeline
        self._mineru_parser = mineru_parser
        self._mineru_writer = mineru_writer
        self.mineru_tier = mineru_tier
        self.mineru_ocr_mode = mineru_ocr_mode
        configure_model_environment(backend, model_home, offline=offline)
        if backend == "mineru" and mineru_small_backend is not None:
            os.environ["MINERU_MODEL_SMALL_BACKEND"] = mineru_small_backend
        if backend == "mineru" and mineru_vlm_engine is not None:
            os.environ["MINERU_MODEL_VLM_ENGINE"] = mineru_vlm_engine

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:
                raise RuntimeError(
                    "OCR dependencies are missing; install with: uv sync --extra ocr"
                ) from exc
            self._pipeline = PaddleOCRVL()
        return self._pipeline

    def _extract_with_paddleocr(self, input_file: Path, document_dir: Path) -> list[Path]:
        results = list(self.pipeline.predict(input=str(input_file)))
        if not results:
            raise RuntimeError(f"OCR returned no result for {input_file}")
        if len(results) > 1:
            results = list(self.pipeline.restructure_pages(results, concatenate_pages=True))

        for result in results:
            result.save_to_markdown(save_path=str(document_dir))

        markdown_files = sorted(document_dir.glob("*.md"))
        if not markdown_files:
            raise RuntimeError(f"OCR did not create Markdown for {input_file}")
        return markdown_files

    def _extract_with_mineru(self, input_file: Path, document_dir: Path) -> list[Path]:
        if self._mineru_parser is None or self._mineru_writer is None:
            try:
                from mineru.parser import parse
                from mineru.parser.writer import FileBasedDataWriter
            except ImportError as exc:
                raise RuntimeError(
                    "MinerU dependencies are missing; install with: uv sync --extra mineru"
                ) from exc
            self._mineru_parser = parse
            self._mineru_writer = FileBasedDataWriter

        result = self._mineru_parser(
            str(input_file),
            tier=self.mineru_tier,
            ocr_mode=self.mineru_ocr_mode,
        )
        result.save(self._mineru_writer(str(document_dir)))
        mineru_markdown = document_dir / "markdown.md"
        if not mineru_markdown.is_file():
            raise RuntimeError(f"MinerU did not create Markdown for {input_file}")
        output_file = document_dir / f"{input_file.stem}.md"
        mineru_markdown.replace(output_file)
        return [output_file]

    def extract_file(self, input_file: Path, output_dir: Path) -> list[Path]:
        if input_file.suffix.lower() not in SUPPORTED_DOCUMENTS:
            raise ValueError(f"unsupported input type: {input_file.suffix or '<none>'}")

        document_dir = output_dir / input_file.stem
        document_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "mineru":
            return self._extract_with_mineru(input_file, document_dir)
        return self._extract_with_paddleocr(input_file, document_dir)

    def extract_path(
        self,
        input_path: Path,
        output_dir: Path,
        recursive: bool = False,
    ) -> list[Path]:
        documents = iter_documents(input_path, recursive=recursive)
        if not documents:
            raise ValueError(f"no supported documents found in {input_path}")

        output_files: list[Path] = []
        root = input_path if input_path.is_dir() else input_path.parent
        for document in documents:
            relative_parent = document.relative_to(root).parent
            output_files.extend(self.extract_file(document, output_dir / relative_parent))
        return output_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-extractor extract",
        description="Extract Markdown and embedded images with PaddleOCR-VL or MinerU.",
    )
    parser.add_argument("input", type=Path, help="input document or directory")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--recursive", action="store_true", help="scan subdirectories")
    parser.add_argument("--backend", choices=EXTRACTION_BACKENDS, default="paddleocr")
    parser.add_argument("--model-home", type=Path, default=Path("models"))
    parser.add_argument("--offline", action="store_true", help="use local models without network")
    parser.add_argument(
        "--mineru-tier",
        choices=MINERU_TIERS,
        default="standard",
    )
    parser.add_argument(
        "--mineru-ocr-mode",
        choices=MINERU_OCR_MODES,
        default="auto",
    )
    parser.add_argument("--mineru-small-backend", choices=MINERU_SMALL_BACKENDS, default="onnx")
    parser.add_argument(
        "--mineru-vlm-engine",
        choices=MINERU_VLM_ENGINES,
        default="llama-cpp",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extractor = DocumentExtractor(
        backend=args.backend,
        model_home=args.model_home,
        offline=args.offline,
        mineru_tier=args.mineru_tier,
        mineru_ocr_mode=args.mineru_ocr_mode,
        mineru_small_backend=args.mineru_small_backend,
        mineru_vlm_engine=args.mineru_vlm_engine,
    )
    outputs = extractor.extract_path(args.input, args.output, args.recursive)
    print(f"Extracted {len(outputs)} Markdown file(s) into {args.output}")
    return 0
