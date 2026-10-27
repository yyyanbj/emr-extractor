from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

DEFAULT_PROMPT = """Extract all relevant information from the medical record as JSON.
Return JSON only, without commentary. Preserve the source language and use null for missing values.
Do not infer facts that are not present in the record."""


def parse_json_response(content: str) -> Any:
    """Parse JSON from a plain or fenced model response."""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else cleaned
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned invalid JSON: {exc.msg}") from exc


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


class RecordStructurer:
    """Turn extracted Markdown into JSON through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434/v1/",
        api_key: str = "openai",
        model: str = "qwen3:8b",
        temperature: float = 0.0,
        prompt: str = DEFAULT_PROMPT,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "LLM dependencies are missing; install with: uv sync --extra llm"
                ) from exc
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client

    def structure_text(self, markdown: str) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": markdown},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned an empty response")
        return parse_json_response(content)

    def structure_file(self, markdown_file: Path, output_file: Path) -> Path:
        markdown = markdown_file.read_text(encoding="utf-8")
        _write_json(output_file, self.structure_text(markdown))
        return output_file


def iter_markdown(path: Path, recursive: bool = False) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() != ".md":
            raise ValueError(f"expected a Markdown file, got: {path}")
        return [path]
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(path.glob(pattern))


def _output_file(source: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_file() and output_path.suffix.lower() == ".json":
        return output_path
    root = input_path if input_path.is_dir() else input_path.parent
    return output_path / source.relative_to(root).with_suffix(".json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-extractor structure",
        description="Convert extracted Markdown records to structured JSON.",
    )
    parser.add_argument("input", type=Path, help="Markdown file or directory")
    parser.add_argument("output", type=Path, help="JSON file or output directory")
    parser.add_argument("--recursive", action="store_true", help="scan subdirectories")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1/")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "openai"))
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = iter_markdown(args.input, recursive=args.recursive)
    if not sources:
        raise ValueError(f"no Markdown files found in {args.input}")

    structurer = RecordStructurer(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        prompt=args.prompt,
    )
    for source in sources:
        structurer.structure_file(source, _output_file(source, args.input, args.output))
    print(f"Structured {len(sources)} record(s) into {args.output}")
    return 0
