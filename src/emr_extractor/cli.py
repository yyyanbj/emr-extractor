from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from emr_extractor import __version__

COMMANDS = {
    "extract": (
        "emr_extractor.extraction",
        "main",
        "extract Markdown and images with PaddleOCR-VL or MinerU",
    ),
    "structure": ("emr_extractor.structuring", "main", "convert Markdown to structured JSON"),
    "redact-fields": (
        "emr_extractor.redaction",
        "fields_main",
        "redact selected fields in structured JSON",
    ),
    "redact-areas": (
        "emr_extractor.redaction",
        "areas_main",
        "irreversibly cover selected PDF or image regions",
    ),
    "redact-detected-fields": (
        "emr_extractor.ocr_redaction",
        "main",
        "detect and cover labelled fields in PDFs with OCR",
    ),
    "download-models": (
        "emr_extractor.models",
        "main",
        "download PaddleOCR and/or MinerU models for offline use",
    ),
}


def _print_help() -> None:
    print("usage: emr-extractor <command> [options]\n")
    print("Extract, structure, and redact scanned medical records.\n")
    print("commands:")
    width = max(map(len, COMMANDS))
    for name, (_, _, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nRun 'emr-extractor <command> --help' for command options.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version"}:
        print(__version__)
        return 0

    command = args.pop(0)
    target = COMMANDS.get(command)
    if target is None:
        print(f"error: unknown command: {command}", file=sys.stderr)
        _print_help()
        return 2

    module_name, function_name, _ = target
    handler = getattr(importlib.import_module(module_name), function_name)
    try:
        result = handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return int(result or 0)
