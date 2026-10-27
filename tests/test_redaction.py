import json

import pymupdf
import pytest
from PIL import Image

from emr_extractor.redaction import (
    DEFAULT_REDACTION_FIELDS,
    fields_parser,
    normalize_box,
    page_areas,
    parse_areas,
    parse_page_selector,
    redact_documents,
    redact_fields,
    redact_json_files,
)


def test_normalize_box_reorders_and_clamps_coordinates():
    assert normalize_box([1.2, 0.8, -0.2, 0.1]) == (0.0, 0.1, 1.0, 0.8)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 1], [0, 0, float("nan"), 1]])
def test_normalize_box_rejects_invalid_areas(box):
    with pytest.raises(ValueError):
        normalize_box(box)


def test_page_selectors_and_all_page_areas():
    assert parse_page_selector("3-1", 4) == [0, 1, 2]
    assert parse_page_selector("1,3,1", 4) == [0, 2]

    config = parse_areas({"ALL": [[0, 0, 0.1, 0.1]], "2": [[0.9, 0.9, 1, 1]]})
    assert len(page_areas(config, 2)[0]) == 1
    assert len(page_areas(config, 2)[1]) == 2


def test_redact_fields_is_recursive_and_does_not_mutate_input():
    source = {
        "Name": "Alice",
        "visits": [{"phone": "123", "diagnosis": "synthetic"}],
    }

    result, count = redact_fields(source, ["name", "PHONE"])

    assert count == 2
    assert result == {
        "Name": "[REDACTED]",
        "visits": [{"phone": "[REDACTED]", "diagnosis": "synthetic"}],
    }
    assert source["Name"] == "Alice"


def test_default_redaction_fields():
    args = fields_parser().parse_args(["input.json", "output.json"])
    assert args.fields == list(DEFAULT_REDACTION_FIELDS)


def test_redact_json_file(tmp_path):
    source = tmp_path / "record.json"
    destination = tmp_path / "safe.json"
    source.write_text(json.dumps({"姓名": "测试姓名", "note": "synthetic"}), encoding="utf-8")

    outputs, count = redact_json_files(source, destination, ["姓名"])

    assert outputs == [destination]
    assert count == 1
    assert json.loads(destination.read_text(encoding="utf-8"))["姓名"] == "[REDACTED]"


def test_redact_image_area(tmp_path):
    source = tmp_path / "record.png"
    destination = tmp_path / "record_redacted.png"
    Image.new("RGB", (20, 10), "white").save(source)

    redact_documents(source, destination, parse_areas([[0, 0, 0.5, 1]]))

    with Image.open(destination) as result:
        assert result.getpixel((2, 5)) == (0, 0, 0)
        assert result.getpixel((18, 5)) == (255, 255, 255)


def test_redact_pdf_area(tmp_path):
    source = tmp_path / "record.pdf"
    destination = tmp_path / "record_redacted.pdf"
    document = pymupdf.open()
    page = document.new_page(width=100, height=100)
    page.draw_rect((0, 0, 100, 100), fill=(1, 1, 1))
    document.save(source)
    document.close()

    redact_documents(source, destination, parse_areas([[0, 0, 0.5, 1]]), dpi=72)

    result = pymupdf.open(destination)
    pixmap = result[0].get_pixmap(dpi=72, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    assert image.getpixel((10, 50)) == (0, 0, 0)
    assert image.getpixel((90, 50)) == (255, 255, 255)
    image.close()
    result.close()
