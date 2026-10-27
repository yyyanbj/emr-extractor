from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pymupdf
from PIL import Image, ImageColor, ImageDraw

Box = tuple[float, float, float, float]
AreaConfig = dict[str, list[Box]]
SUPPORTED_DOCUMENTS = {".pdf", ".jpg", ".jpeg", ".png"}
DEFAULT_REDACTION_FIELDS = ("Name", "Phone", "Address", "Doctor")


def normalize_box(values: Iterable[float]) -> Box:
    coordinates = tuple(float(value) for value in values)
    if len(coordinates) != 4:
        raise ValueError("each area must contain exactly four coordinates")
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("area coordinates must be finite numbers")

    x1, y1, x2, y2 = (max(0.0, min(1.0, value)) for value in coordinates)
    box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if box[0] == box[2] or box[1] == box[3]:
        raise ValueError("area must have a non-zero width and height")
    return box


def parse_areas(value: str | list[Any] | dict[str, Any]) -> AreaConfig:
    """Parse a list applied to all pages or a page-selector mapping."""
    raw = json.loads(value) if isinstance(value, str) else value
    if isinstance(raw, list):
        raw = {"all": raw}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("areas must be a non-empty JSON list or object")

    parsed: AreaConfig = {}
    for selector, boxes in raw.items():
        if not isinstance(boxes, list):
            raise ValueError(f"areas for page selector {selector!r} must be a list")
        key = str(selector).strip()
        parsed["all" if key.lower() == "all" else key] = [normalize_box(box) for box in boxes]
    if not any(parsed.values()):
        raise ValueError("at least one redaction area is required")
    return parsed


def parse_page_selector(selector: str, total_pages: int) -> list[int]:
    selector = selector.strip().lower()
    if selector == "all":
        return list(range(total_pages))
    if "," in selector:
        pages: list[int] = []
        for part in selector.split(","):
            if part.strip():
                pages.extend(parse_page_selector(part, total_pages))
        return list(dict.fromkeys(pages))
    if "-" in selector:
        start_text, end_text = (part.strip() for part in selector.split("-", 1))
        if not start_text.isdigit() or not end_text.isdigit():
            raise ValueError(f"invalid page selector: {selector}")
        start, end = sorted((int(start_text), int(end_text)))
        if start < 1:
            raise ValueError("page numbers start at 1")
        return [page - 1 for page in range(start, end + 1) if page <= total_pages]
    if selector.isdigit():
        page = int(selector)
        if page < 1:
            raise ValueError("page numbers start at 1")
        return [page - 1] if page <= total_pages else []
    raise ValueError(f"invalid page selector: {selector}")


def page_areas(config: AreaConfig, total_pages: int) -> dict[int, list[Box]]:
    areas = {page: list(config.get("all", [])) for page in range(total_pages)}
    for selector, boxes in config.items():
        if selector.strip().lower() == "all":
            continue
        for page in parse_page_selector(selector, total_pages):
            areas[page].extend(boxes)
    return {page: boxes for page, boxes in areas.items() if boxes}


def apply_areas(image: Image.Image, boxes: Sequence[Box], fill: str = "black") -> Image.Image:
    try:
        color = ImageColor.getcolor(fill, image.mode)
    except ValueError as exc:
        raise ValueError(f"invalid fill color: {fill}") from exc

    draw = ImageDraw.Draw(image)
    width, height = image.size
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            (round(width * x1), round(height * y1), round(width * x2), round(height * y2)),
            fill=color,
        )
    return image


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=destination.suffix, delete=False
    ) as handle:
        return Path(handle.name)


def redact_pdf(
    source: Path,
    destination: Path,
    config: AreaConfig,
    *,
    dpi: int = 300,
    fill: str = "black",
) -> None:
    if dpi < 1:
        raise ValueError("dpi must be greater than zero")
    if source.resolve() == destination.resolve():
        raise ValueError("refusing to overwrite the source document")

    document = pymupdf.open(source)
    images: list[Image.Image] = []
    temporary: Path | None = None
    try:
        masks = page_areas(config, document.page_count)
        if not masks:
            raise ValueError(f"no configured area applies to {source}")
        for page_number in range(document.page_count):
            pixmap = document.load_page(page_number).get_pixmap(dpi=dpi, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            apply_areas(image, masks.get(page_number, []), fill)
            images.append(image)
        if not images:
            raise ValueError(f"empty PDF: {source}")

        temporary = _temporary_path(destination)
        images[0].save(temporary, "PDF", save_all=True, append_images=images[1:])
        temporary.replace(destination)
    finally:
        document.close()
        for image in images:
            image.close()
        if temporary and temporary.exists():
            temporary.unlink()


def redact_image(
    source: Path,
    destination: Path,
    config: AreaConfig,
    *,
    fill: str = "black",
) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("refusing to overwrite the source image")
    masks = page_areas(config, 1).get(0, [])
    if not masks:
        raise ValueError(f"no configured area applies to {source}")

    temporary: Path | None = None
    try:
        with Image.open(source) as original:
            image = original.convert("RGB")
            apply_areas(image, masks, fill)
            temporary = _temporary_path(destination)
            image.save(temporary)
            image.close()
        temporary.replace(destination)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def iter_documents(path: Path, recursive: bool = False) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_DOCUMENTS:
            raise ValueError(f"unsupported input type: {path.suffix or '<none>'}")
        return [path]
    candidates = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        item for item in candidates if item.is_file() and item.suffix.lower() in SUPPORTED_DOCUMENTS
    )


def _document_output(source: Path, input_path: Path, output_path: Path, suffix: str) -> Path:
    if input_path.is_file() and output_path.suffix.lower() in SUPPORTED_DOCUMENTS:
        return output_path
    root = input_path if input_path.is_dir() else input_path.parent
    relative = source.relative_to(root)
    return output_path / relative.parent / f"{source.stem}{suffix}{source.suffix.lower()}"


def redact_documents(
    input_path: Path,
    output_path: Path,
    config: AreaConfig,
    *,
    recursive: bool = False,
    suffix: str = "_redacted",
    dpi: int = 300,
    fill: str = "black",
) -> list[Path]:
    documents = iter_documents(input_path, recursive)
    if not documents:
        raise ValueError(f"no supported documents found in {input_path}")

    outputs: list[Path] = []
    for source in documents:
        destination = _document_output(source, input_path, output_path, suffix)
        if source.suffix.lower() == ".pdf":
            redact_pdf(source, destination, config, dpi=dpi, fill=fill)
        else:
            redact_image(source, destination, config, fill=fill)
        outputs.append(destination)
    return outputs


def redact_fields(
    data: Any, fields: Iterable[str], replacement: Any = "[REDACTED]"
) -> tuple[Any, int]:
    """Return a copy with matching object keys redacted recursively."""
    names = {field.strip().casefold() for field in fields if field.strip()}
    if not names:
        raise ValueError("at least one field name is required")

    count = 0

    def visit(value: Any) -> Any:
        nonlocal count
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key.strip().casefold() in names:
                    result[key] = replacement
                    count += 1
                else:
                    result[key] = visit(child)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(data), count


def _json_output(source: Path, input_path: Path, output_path: Path, suffix: str) -> Path:
    if input_path.is_file() and output_path.suffix.lower() == ".json":
        return output_path
    root = input_path if input_path.is_dir() else input_path.parent
    relative = source.relative_to(root)
    return output_path / relative.parent / f"{source.stem}{suffix}.json"


def redact_json_files(
    input_path: Path,
    output_path: Path,
    fields: Iterable[str],
    *,
    replacement: Any = "[REDACTED]",
    recursive: bool = False,
    suffix: str = "_redacted",
) -> tuple[list[Path], int]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        sources = [input_path]
    else:
        sources = sorted(input_path.glob("**/*.json" if recursive else "*.json"))
    if not sources:
        raise ValueError(f"no JSON files found in {input_path}")

    total = 0
    outputs: list[Path] = []
    for source in sources:
        destination = _json_output(source, input_path, output_path, suffix)
        if source.resolve() == destination.resolve():
            raise ValueError("refusing to overwrite a source JSON file")
        data = json.loads(source.read_text(encoding="utf-8"))
        redacted, count = redact_fields(data, fields, replacement)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(destination)
        try:
            temporary.write_text(
                json.dumps(redacted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        total += count
        outputs.append(destination)
    return outputs, total


def areas_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-extractor redact-areas",
        description="Irreversibly rasterize and cover regions in PDF/JPG/PNG records.",
    )
    parser.add_argument("input", type=Path, help="input document or directory")
    parser.add_argument("output", type=Path, help="output document or directory")
    parser.add_argument(
        "--areas",
        required=True,
        help='normalized JSON boxes, e.g. [[0,0,0.4,0.1]] or {"1":[[...]]}',
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--suffix", default="_redacted")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fill", default="black")
    return parser


def areas_main(argv: Sequence[str] | None = None) -> int:
    args = areas_parser().parse_args(argv)
    outputs = redact_documents(
        args.input,
        args.output,
        parse_areas(args.areas),
        recursive=args.recursive,
        suffix=args.suffix,
        dpi=args.dpi,
        fill=args.fill,
    )
    print(f"Redacted {len(outputs)} document(s) into {args.output}")
    return 0


def fields_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emr-extractor redact-fields",
        description="Redact selected keys recursively in structured JSON records.",
    )
    parser.add_argument("input", type=Path, help="input JSON file or directory")
    parser.add_argument("output", type=Path, help="output JSON file or directory")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_REDACTION_FIELDS),
        help="field names to redact (default: Name Phone Address Doctor)",
    )
    parser.add_argument("--replacement", default="[REDACTED]")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--suffix", default="_redacted")
    return parser


def fields_main(argv: Sequence[str] | None = None) -> int:
    args = fields_parser().parse_args(argv)
    outputs, count = redact_json_files(
        args.input,
        args.output,
        args.fields,
        replacement=args.replacement,
        recursive=args.recursive,
        suffix=args.suffix,
    )
    print(f"Redacted {count} field(s) across {len(outputs)} JSON file(s)")
    return 0
