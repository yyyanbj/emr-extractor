"""Prepare local model caches for online and offline deployments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

MODEL_BACKENDS = ("paddleocr", "mineru", "all")


def configure_model_environment(
    backend: str,
    model_home: Path | None = None,
    *,
    offline: bool = False,
) -> None:
    """Point a backend at a portable model directory before importing it."""
    if backend not in MODEL_BACKENDS:
        raise ValueError(f"unsupported model backend: {backend}")

    if model_home is not None:
        model_home = model_home.expanduser().resolve()
        if backend in {"paddleocr", "all"}:
            paddle_home = model_home / "paddleocr"
            paddle_home.mkdir(parents=True, exist_ok=True)
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_home)
        if backend in {"mineru", "all"}:
            mineru_home = model_home / "mineru"
            mineru_home.mkdir(parents=True, exist_ok=True)
            os.environ["MINERU_HOME"] = str(mineru_home)

    if not offline:
        return
    if backend in {"paddleocr", "all"}:
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if backend in {"mineru", "all"}:
        os.environ["MINERU_MODEL_SOURCE"] = "local"


def download_paddleocr_models(model_home: Path, source: str = "huggingface") -> None:
    configure_model_environment("paddleocr", model_home)
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = source
    try:
        from paddleocr import PaddleOCR, PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR dependencies are missing; install with: uv sync --extra ocr"
        ) from exc

    print("Downloading PaddleOCR-VL and field-detection models...")
    PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device="cpu",
    )
    PaddleOCRVL(device="cpu")

    cache = Path(os.environ["PADDLE_PDX_CACHE_HOME"])
    if not any(path.is_file() for path in cache.rglob("*")):
        raise RuntimeError(f"PaddleOCR model cache is empty: {cache}")
    print(f"PaddleOCR models ready in {cache}")


def _run_mineru(command: list[str]) -> None:
    executable = shutil.which("mineru-kit")
    if executable is None:
        raise RuntimeError("MinerU dependencies are missing; install with: uv sync --extra mineru")
    command[0] = executable
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MinerU command failed with exit code {exc.returncode}") from exc


def download_mineru_models(
    model_home: Path,
    *,
    tier: str = "standard",
    source: str = "auto",
    small_backend: str = "onnx",
    vlm_engine: str = "llama-cpp",
) -> None:
    configure_model_environment("mineru", model_home)
    selection = ["--tier", tier, "--small-backend", small_backend]
    if tier == "standard":
        selection.extend(["--vlm-engine", vlm_engine])
    download_options = list(selection)
    if source != "auto":
        download_options.extend(["--source", source])

    print("Downloading MinerU models...")
    _run_mineru(["mineru-kit", "models", "download", *download_options])
    _run_mineru(["mineru-kit", "models", "verify", *selection])
    print(f"MinerU models ready in {Path(os.environ['MINERU_HOME']) / 'models'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-extractor download-models",
        description="Download and verify local models for an online or offline deployment.",
    )
    parser.add_argument("--backend", choices=MODEL_BACKENDS, default="paddleocr")
    parser.add_argument("--model-home", type=Path, default=Path("models"))
    parser.add_argument("--paddle-source", choices=("huggingface", "bos"), default="huggingface")
    parser.add_argument("--mineru-tier", choices=("basic", "standard"), default="standard")
    parser.add_argument(
        "--mineru-source",
        choices=("auto", "huggingface", "modelscope"),
        default="auto",
    )
    parser.add_argument("--mineru-small-backend", choices=("onnx", "torch"), default="onnx")
    parser.add_argument(
        "--mineru-vlm-engine",
        choices=("llama-cpp", "vllm", "lmdeploy"),
        default="llama-cpp",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend in {"paddleocr", "all"}:
        download_paddleocr_models(args.model_home, args.paddle_source)
    if args.backend in {"mineru", "all"}:
        download_mineru_models(
            args.model_home,
            tier=args.mineru_tier,
            source=args.mineru_source,
            small_backend=args.mineru_small_backend,
            vlm_engine=args.mineru_vlm_engine,
        )
    return 0
