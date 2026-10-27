"""OCR-assisted field redaction for scanned PDFs.

This module preserves the project's layout-specific detector. Prefer the
deterministic ``redact-areas`` command when document layouts are known.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from emr_extractor.models import configure_model_environment
from emr_extractor.redaction import DEFAULT_REDACTION_FIELDS

# =========================
# Types & Utils
# =========================

Box = tuple[float, float, float, float]
DEFAULT_FIELD_KEYWORDS = list(DEFAULT_REDACTION_FIELDS)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def norm_box_ratio(x1: float, y1: float, x2: float, y2: float) -> Box:
    return (
        clamp01(min(x1, x2)),
        clamp01(min(y1, y2)),
        clamp01(max(x1, x2)),
        clamp01(max(y1, y2)),
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_pdfs(input_dir: Path, recursive: bool):
    if recursive:
        yield from input_dir.rglob("*.pdf")
    else:
        yield from input_dir.glob("*.pdf")


# =========================
# OCR page title detection
# =========================


def expand_polygon(poly: list, expand_ratio: float = 0.5) -> list:
    """
    Expand a polygon (list of points) outward from its center.

    Args:
        poly: List of (x, y) points
        expand_ratio: How much to expand (0.5 = expand by 50%)

    Returns:
        Expanded polygon as list of (x, y) tuples
    """
    if len(poly) < 3:
        return poly

    # Calculate center
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # Expand each point away from center
    expanded = []
    for x, y in poly:
        dx = x - cx
        dy = y - cy
        new_x = cx + dx * (1 + expand_ratio)
        new_y = cy + dy * (1 + expand_ratio)
        expanded.append((new_x, new_y))

    return expanded


def detect_sensitive_fields(
    img: Image.Image,
    pipeline,
    search_ratio: float = 1.0,
    extracted_fields: dict[str, str | None] | None = None,
    field_keywords: list[str] | None = None,
) -> list[tuple[str, object]]:
    """
    Use OCR to detect labelled sensitive fields and return all matches.
    Masks the value following each requested label. Supports both rectangular and
    skewed (polygonal) masks.
    Scans entire page by default (search_ratio=1.0) to catch name fields anywhere.

    Args:
        img: Page image
        pipeline: PaddleOCR pipeline (returns list of recognition results)

    Returns:
        List of (value, mask) where mask can be:
        - Box (x1, y1, x2, y2) in 0-1 normalized ratio coordinates
        - Polygon [(x1,y1), (x2,y2), ...] in 0-1 normalized ratio coordinates
    """
    import numpy as np

    w, h_full = img.size
    crop_h = int(h_full * search_ratio)
    crop_h = max(crop_h, 1)
    img_cropped = img.crop((0, 0, w, crop_h))
    img_array = np.array(img_cropped)

    # Run OCR on cropped page (top search_ratio, default 25%)
    # PaddleOCR new API: use predict() instead of ocr()
    results = pipeline.predict(img_array)

    if not results or not results[0]:
        return []

    # Flatten and process results
    # New API returns list of dicts with 'dt_polys' and 'rec_texts'
    text_items = []

    for page_result in results:
        if isinstance(page_result, dict):
            # New format: {'dt_polys': [...], 'rec_texts': [...], 'rec_scores': [...]}
            dt_polys = page_result.get("dt_polys", [])
            rec_texts = page_result.get("rec_texts", [])
            rec_scores = page_result.get("rec_scores", [])

            for i, (poly, text) in enumerate(zip(dt_polys, rec_texts)):
                if poly is not None and len(poly) >= 2:
                    # Store both bounding box and polygon
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    score = rec_scores[i] if i < len(rec_scores) else 0.0

                    text_items.append(
                        {
                            "text": text,
                            "confidence": score,
                            "box": (x1, y1, x2, y2),  # pixel coordinates
                            "poly": poly,  # original polygon from OCR
                        }
                    )
        elif isinstance(page_result, list):
            # Old format compatibility: [[coords, (text, confidence)], ...]
            for line in page_result:
                if isinstance(line, list) and len(line) >= 2:
                    coords = line[0] if isinstance(line[0], list) else []
                    text_info = line[1] if isinstance(line[1], (list, tuple)) else (line[1], 0.0)
                    text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                    confidence = (
                        text_info[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) > 1
                        else 0.0
                    )

                    if coords and len(coords) >= 2:
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        text_items.append(
                            {
                                "text": text,
                                "confidence": confidence,
                                "box": (x1, y1, x2, y2),
                                "poly": coords,
                            }
                        )

    # Look for name and personal info keywords to mask
    # Do NOT treat generic 'optom' alone as a keyword (avoid masking 'Optometry Clinic')
    name_keywords = [
        keyword.strip() for keyword in (field_keywords or DEFAULT_FIELD_KEYWORDS) if keyword.strip()
    ]

    # For mask expansion: name-related fields keep 1 + 0.8 height; others use 0.9
    name_core_keywords = {"name:", "姓名:", "name", "姓名", "patient name:", "patient name"}
    name_core_keywords = {kw.lower() for kw in name_core_keywords}

    matches: list[tuple[str, Box]] = []

    # Special case: if "BINOCULAR" and "Name:" both present, mask the area after "Name:"
    # This handles cases where Name: is followed by empty box or non-name text
    binocular_idx = None
    name_label_idx = None
    for idx, item in enumerate(text_items):
        text_lower = item["text"].lower().strip()
        if "binocular" in text_lower:
            binocular_idx = idx
        if text_lower in ["name:", "name", "姓名:", "姓名"]:
            name_label_idx = idx

    requested_names = {keyword.lower().rstrip(":") for keyword in name_keywords}
    if (
        requested_names.intersection({"name", "patient name", "姓名"})
        and binocular_idx is not None
        and name_label_idx is not None
    ):
        # Create a mask area to the right of the Name: label
        # Extend right by ~3x the name label width
        name_label_item = text_items[name_label_idx]
        name_box = name_label_item["box"]
        x1, y1, x2, y2 = name_box

        # Calculate width of "Name:" label
        name_label_width = x2 - x1

        # Create mask: start from x of name label, extend right by 3x width
        x1_mask = x2
        y1_mask = y1
        x2_mask = min(w, x2 + name_label_width * 3)  # Extend right by 3x name width
        y2_mask = y2  # Keep same height as name label

        # Normalize
        x1_norm = x1_mask / w
        y1_norm = y1_mask / h_full
        x2_norm = x2_mask / w
        y2_norm = y2_mask / h_full
        norm_box = norm_box_ratio(x1_norm, y1_norm, x2_norm, y2_norm)

        matches.append(("(Name field area)", norm_box))

    for i, item in enumerate(text_items):
        text = item["text"].strip()
        text_lower = text.lower().strip()

        # Check if this text contains a name keyword
        # Case 1: "Name: John Doe" or "Staff optom: xxx" - keyword and value in same OCR box
        matched_kw = None
        matched_kw_len = 0
        for kw in name_keywords:
            kw_lower = kw.lower()
            # Match if:
            # 1. Text starts with keyword and followed by space/colon/end
            if text_lower.startswith(kw_lower):
                end_pos = len(kw_lower)
                if end_pos >= len(text_lower) or text_lower[end_pos] in [" ", ":"]:
                    matched_kw = kw_lower
                    matched_kw_len = len(kw)
                    break
            # 2. Keyword as separate word (with space before and space/colon after)
            elif f" {kw_lower} " in text_lower or f" {kw_lower}:" in text_lower:
                matched_kw = kw_lower
                matched_kw_len = len(kw)
                break

        if matched_kw:
            # Extract the value part after the keyword
            kw_pos = text_lower.find(matched_kw)
            if kw_pos == -1:
                continue
            value_start = kw_pos + matched_kw_len
            if value_start < len(text):
                name_value = text[value_start:].strip()
                # Remove leading colon/space if present
                name_value = name_value.lstrip(":").strip()

                # Skip empty value (e.g., "Staff:" with no name)
                if not name_value:
                    continue

                # Filter: must have actual content, not too short, not too long
                if 2 < len(name_value) < 200 and name_value not in [":", "."]:
                    # Choose vertical adjustments:
                    # - name fields: expand downward by 0.8h (keep y1), widen horizontally
                    # - other fields: shrink upward by 0.1h and shrink downward by 0.1h
                    is_name_field = matched_kw in name_core_keywords
                    down_factor = 0.8 if is_name_field else -0.1

                    # Try to use polygon if available, otherwise fall back to box
                    if "poly" in item and item["poly"] is not None and len(item["poly"]) > 0:
                        # Use OCR polygon (skewed/rotated) - convert to normalized coords
                        try:
                            poly = item["poly"]
                            # Expand polygon slightly to ensure complete coverage
                            expanded_poly = expand_polygon(poly, expand_ratio=0.3)
                            # Normalize to 0-1 range
                            norm_poly = [(x / w, y / h_full) for x, y in expanded_poly]
                            matches.append((name_value, norm_poly))

                            # For name fields, add Chinese name mask below (polygon case)
                            if is_name_field:
                                has_other_fields = False
                                if extracted_fields:
                                    key_fields = ["File No.", "DoB", "Gender", "Latest Appt"]
                                    for field in key_fields:
                                        if extracted_fields.get(field) is not None:
                                            has_other_fields = True
                                            break

                                if has_other_fields:
                                    # Get bounding box from polygon for Chinese name positioning
                                    poly_xs = [pt[0] for pt in expanded_poly]
                                    poly_ys = [pt[1] for pt in expanded_poly]
                                    poly_x1, poly_x2 = min(poly_xs), max(poly_xs)
                                    poly_y1, poly_y2 = min(poly_ys), max(poly_ys)
                                    poly_height = poly_y2 - poly_y1

                                    # Chinese name parallel with English, below with small gap
                                    chinese_y1 = min(h_full, poly_y2 + poly_height * 0.1)
                                    chinese_y2 = min(h_full, chinese_y1 + poly_height)

                                    # Normalize
                                    c_x1_norm = poly_x1 / w
                                    c_y1_norm = chinese_y1 / h_full
                                    c_x2_norm = poly_x2 / w
                                    c_y2_norm = chinese_y2 / h_full
                                    chinese_name_box = norm_box_ratio(
                                        c_x1_norm, c_y1_norm, c_x2_norm, c_y2_norm
                                    )
                                    matches.append(("(Chinese name)", chinese_name_box))

                            continue
                        except Exception:
                            pass  # Fall back to box if polygon expansion fails

                    # Fall back to rectangular mask
                    # Use this box as the mask area, expand it 1.8x
                    pixel_box = item["box"]
                    x1, y1, x2, y2 = pixel_box

                    # IMPORTANT: The OCR box includes both keyword and the value
                    # Adjust x1 to only mask the value part
                    total_text = text.strip()
                    value_proportion = (
                        len(name_value) / len(total_text) if len(total_text) > 0 else 1.0
                    )
                    estimated_value_start_x = x1 + (1.0 - value_proportion) * (x2 - x1)
                    x1_adjusted = max(x1, estimated_value_start_x)

                    height = y2 - y1
                    x1_expanded = max(x1_adjusted - height * 0.5, 0)  # Expand left a bit from value
                    x2_expanded = min(w, x2 + height * 0.5)  # Expand right a bit

                    if is_name_field:
                        y1_expanded = y1  # keep top
                        y2_expanded = min(h_full, y2 + height * down_factor)
                    else:
                        # shrink: move top down and bottom up by 0.1h
                        y1_expanded = min(h_full, y1 + height * 0.1)
                        y2_expanded = max(0, y2 + height * down_factor)

                    # Normalize using full page height
                    x1_norm = x1_expanded / w
                    y1_norm = y1_expanded / h_full
                    x2_norm = x2_expanded / w
                    y2_norm = y2_expanded / h_full
                    norm_box = norm_box_ratio(x1_norm, y1_norm, x2_norm, y2_norm)
                    matches.append((name_value, norm_box))

                    # For name fields, add additional mask below for Chinese name
                    # Add the second-line name only when the surrounding form was detected.
                    if is_name_field:
                        has_other_fields = False
                        if extracted_fields:
                            key_fields = ["File No.", "DoB", "Gender", "Latest Appt"]
                            for field in key_fields:
                                if extracted_fields.get(field) is not None:
                                    has_other_fields = True
                                    break

                        # Only add Chinese name mask if multiple fields detected (indicates a form)
                        if has_other_fields:
                            # Chinese name has same x-range as English name (parallel)
                            chinese_name_x1 = x1_expanded
                            chinese_name_x2 = x2_expanded  # Use same x-range as English name
                            # Chinese name is on next line below English name
                            chinese_name_y1 = min(h_full, y2_expanded + height * 0.1)  # Small gap
                            chinese_name_y2 = min(
                                h_full, chinese_name_y1 + height
                            )  # Same height as English

                            # Normalize
                            c_x1_norm = chinese_name_x1 / w
                            c_y1_norm = chinese_name_y1 / h_full
                            c_x2_norm = chinese_name_x2 / w
                            c_y2_norm = chinese_name_y2 / h_full
                            chinese_name_box = norm_box_ratio(
                                c_x1_norm, c_y1_norm, c_x2_norm, c_y2_norm
                            )
                            matches.append(("(Chinese name)", chinese_name_box))

                    continue

        # Case 2: "Name:" in one box, "John Doe" in next box
        label_match_kw = None
        for kw in name_keywords:
            kw_lower = kw.lower()
            if text_lower == kw_lower or text_lower == kw_lower.rstrip(":") + ":":
                label_match_kw = kw_lower
                break

        if label_match_kw:
            # Found the label, try to get the next item as the value
            # Keep searching until we find a non-empty value or run out of items
            for next_idx in range(i + 1, len(text_items)):
                next_item = text_items[next_idx]
                name_value = next_item["text"].strip()

                # Skip if value too short/long or contains colon
                if 2 < len(name_value) < 200 and ":" not in name_value:
                    # Try to use polygon if available, otherwise fall back to box
                    if (
                        "poly" in next_item
                        and next_item["poly"] is not None
                        and len(next_item["poly"]) > 0
                    ):
                        # Use OCR polygon (skewed/rotated) - convert to normalized coords
                        try:
                            poly = next_item["poly"]
                            # Expand polygon slightly to ensure complete coverage
                            expanded_poly = expand_polygon(poly, expand_ratio=0.3)
                            # Normalize to 0-1 range
                            norm_poly = [(x / w, y / h_full) for x, y in expanded_poly]
                            matches.append((name_value, norm_poly))
                            break
                        except Exception:
                            pass  # Fall back to box if polygon expansion fails

                    # Fall back to rectangular mask
                    pixel_box = next_item["box"]  # (x1, y1, x2, y2) in pixels
                    x1, y1, x2, y2 = pixel_box

                    # Vertical adjustments: names expand down; others shrink up/down by 0.1h
                    is_name_field = label_match_kw in name_core_keywords
                    down_factor = 0.8 if is_name_field else -0.1
                    height = y2 - y1
                    x1_expanded = max(x1, x1 - height * 0.5)
                    x2_expanded = min(w, x2 + height * 0.5)

                    if is_name_field:
                        y1_expanded = y1
                        y2_expanded = min(h_full, y2 + height * down_factor)
                    else:
                        y1_expanded = min(h_full, y1 + height * 0.1)
                        y2_expanded = max(0, y2 + height * down_factor)

                    # Convert to normalized coordinates
                    x1_norm = x1_expanded / w
                    y1_norm = y1_expanded / h_full
                    x2_norm = x2_expanded / w
                    y2_norm = y2_expanded / h_full

                    norm_box = norm_box_ratio(x1_norm, y1_norm, x2_norm, y2_norm)
                    matches.append((name_value, norm_box))

                    # For name fields, add additional mask below for Chinese name
                    # Add the second-line name only when the surrounding form was detected.
                    if is_name_field:
                        has_other_fields = False
                        if extracted_fields:
                            key_fields = ["File No.", "DoB", "Gender", "Latest Appt"]
                            for field in key_fields:
                                if extracted_fields.get(field) is not None:
                                    has_other_fields = True
                                    break

                        # Only add Chinese name mask if multiple fields detected (indicates a form)
                        if has_other_fields:
                            # Chinese name has same x-range as English name (parallel)
                            chinese_name_x1 = x1_expanded
                            chinese_name_x2 = x2_expanded  # Use same x-range as English name
                            # Chinese name is on next line below English name
                            chinese_name_y1 = min(h_full, y2_expanded + height * 0.1)  # Small gap
                            chinese_name_y2 = min(
                                h_full, chinese_name_y1 + height
                            )  # Same height as English

                            # Normalize
                            c_x1_norm = chinese_name_x1 / w
                            c_y1_norm = chinese_name_y1 / h_full
                            c_x2_norm = chinese_name_x2 / w
                            c_y2_norm = chinese_name_y2 / h_full
                            chinese_name_box = norm_box_ratio(
                                c_x1_norm, c_y1_norm, c_x2_norm, c_y2_norm
                            )
                            matches.append(("(Chinese name)", chinese_name_box))

                    break
                elif name_value:
                    # Non-empty but fails validation, stop searching
                    break
                else:
                    # Empty value, continue to next
                    continue

    return matches


def extract_key_fields(
    img: Image.Image,
    pipeline,
    search_ratio: float = 0.35,
) -> dict[str, str | None]:
    """
    Extract key fields from OCR results: DoB, Gender, Latest Appt, File No.

    Args:
        img: Page image
        pipeline: PaddleOCR pipeline
        search_ratio: Proportion of page to search (default 35% from top)

    Returns:
        Dict with keys: 'DoB', 'Gender', 'Latest Appt', 'File No.'
    """
    import numpy as np

    w, h_full = img.size
    crop_h = int(h_full * search_ratio)
    crop_h = max(crop_h, 1)
    img_cropped = img.crop((0, 0, w, crop_h))
    img_array = np.array(img_cropped)

    # Run OCR on cropped page
    results = pipeline.predict(img_array)

    extracted_fields = {
        "DoB": None,
        "Gender": None,
        "Latest Appt": None,
        "File No.": None,
    }

    if not results:
        return extracted_fields

    # Parse OCR results - new API returns list with dict containing 'rec_texts'
    text_items = []

    for page_result in results:
        if isinstance(page_result, dict):
            # New format from PaddleX: {'dt_polys': [...], 'rec_texts': [...], 'rec_scores': [...]}
            rec_texts = page_result.get("rec_texts", [])
            for text in rec_texts:
                if text:
                    text_items.append(str(text).strip())
        elif isinstance(page_result, list):
            # Old format compatibility: [[coords, (text, confidence)], ...]
            for line in page_result:
                if isinstance(line, list) and len(line) >= 2:
                    text_info = line[1] if isinstance(line[1], (list, tuple)) else (line[1], 0.0)
                    text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                    if text:
                        text_items.append(str(text).strip())

    if not text_items:
        return extracted_fields

    # Join all text to make searching easier
    full_text = " ".join(text_items)

    # Extract Date of Birth (DoB)
    # Pattern: "DoB" or "D.O.B" or "Date of Birth" followed by a date
    dob_patterns = [
        r"(?:dob|d\.o\.b|d\.?o\.?b\.?|date\s+of\s+birth)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(?:dob|d\.o\.b|d\.?o\.?b\.?|date\s+of\s+birth)[\s:]*(\d{4}-\d{1,2}-\d{1,2})",
        r"(?:dob|d\.o\.b|d\.?o\.?b\.?|date\s+of\s+birth)[\s:]*(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted_fields["DoB"] = match.group(1).strip()
            break

    # Extract Gender (M/F or Male/Female)
    gender_patterns = [
        r"(?:gender|sex)[\s:]*\s*([MFmf](?:ale|emale)?)",
        r"(?:gender|sex)[\s:]*([MFmf])",
    ]
    for pattern in gender_patterns:
        gender_match = re.search(pattern, full_text, re.IGNORECASE)
        if gender_match:
            gender_text = gender_match.group(1).strip().upper()
            if gender_text.startswith("M"):
                extracted_fields["Gender"] = "M"
                break
            elif gender_text.startswith("F"):
                extracted_fields["Gender"] = "F"
                break

    # Extract Latest Appointment
    # Pattern: "Latest Appt" or "Last Appt" followed by a date
    latest_appt_patterns = [
        r"(?:latest\s+appt|last\s+appt|last\s+appointment)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        r"(?:latest\s+appt|last\s+appt|last\s+appointment)[\s:]*(\d{4}-\d{1,2}-\d{1,2})",
        r"(?:latest\s+appt|last\s+appt|last\s+appointment)[\s:]*(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})",
    ]
    for pattern in latest_appt_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted_fields["Latest Appt"] = match.group(1).strip()
            break

    # Extract File No. (Pattern: P followed by digits like P123456)
    file_no_match = re.search(r"(?:file\s+no|file\s*#|file\s+number)[\s:]*([Pp]\d+)", full_text)
    if not file_no_match:
        # Also try to find just "P" followed by digits (common pattern)
        file_no_match = re.search(r"\b([Pp]\d{6,})\b", full_text)

    if file_no_match:
        extracted_fields["File No."] = file_no_match.group(1).strip()

    return extracted_fields


def detect_page_title(
    img: Image.Image,
    pipeline,
    title_keywords: dict[str, list[str]],
    search_region: Box | None = None,
) -> str | None:
    """
    Use OCR to detect page title/type based on keywords.

    Args:
        img: Page image
        pipeline: PaddleX OCR pipeline
        title_keywords: Dict mapping title names to keyword lists
        search_region: Optional region to search (x1,y1,x2,y2 in 0-1 ratio)
                      Default: top 30% of page

    Returns:
        Detected title name or None
    """
    import numpy as np

    if search_region is None:
        search_region = (0.0, 0.0, 1.0, 0.25)  # Top 25% by default

    # Crop to search region
    w, h = img.size
    x1 = int(w * search_region[0])
    y1 = int(h * search_region[1])
    x2 = int(w * search_region[2])
    y2 = int(h * search_region[3])

    search_img = img.crop((x1, y1, x2, y2))

    # Convert PIL Image to numpy array (PaddleOCR requires numpy.ndarray)
    img_array = np.array(search_img)

    # Run OCR
    results = pipeline.predict(img_array)

    # Extract all text
    all_text = ""
    if results:
        for line in results:
            if line:
                for word_info in line["rec_texts"]:
                    text = word_info
                    all_text += text + " "

    all_text = all_text.lower()

    if "Supplementary Record".lower() in all_text:
        return "flag"

    # Match keywords to determine title
    for title, keywords in title_keywords.items():
        for keyword in keywords:
            if keyword.lower() in all_text:
                return title

    return None


# =========================
# Rendering
# =========================


def rasterize_page(page: pymupdf.Page, dpi: int, target_width: int = 1600) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Resize to fixed width for faster OCR processing
    if img.width > target_width:
        scale_ratio = target_width / img.width
        new_height = int(img.height * scale_ratio)
        img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

    return img


def apply_masks(img: Image.Image, masks, fill: str) -> Image.Image:
    """
    Apply masks to image. Masks can be:
    - List of Box (rectangles): [(x1,y1,x2,y2), ...]
    - List of polygons: [[(x1,y1), (x2,y2), ...], ...]
    - Mixed: list of either Box or polygon
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for mask in masks:
        if isinstance(mask, (list, tuple)):
            # Check if it's a Box (4 elements) or polygon (multiple points)
            if len(mask) == 4 and all(isinstance(x, (int, float)) for x in mask):
                # Rectangle: (x1, y1, x2, y2)
                x1, y1, x2, y2 = mask
                draw.rectangle(
                    [int(w * x1), int(h * y1), int(w * x2), int(h * y2)],
                    fill=fill,
                )
            elif len(mask) >= 3:
                # Polygon: list of (x, y) tuples
                # Convert normalized coords to pixel coords
                pixel_points = [(int(w * x), int(h * y)) for x, y in mask]
                draw.polygon(pixel_points, fill=fill)

    return img


# =========================
# Core redaction
# =========================


def redact_pdf_with_ocr(
    in_pdf: Path,
    out_pdf: Path,
    dpi: int,
    fill: str,
    ocr_pipeline,
    title_keywords: dict[str, list[str]],
    title_masks: dict[str, list[Box]],
    search_region: Box | None,
    default_masks: list[Box] | None,
    field_keywords: list[str] | None = None,
    detect_fields_flag: bool = True,
    extract_fields_flag: bool = False,
    verbose: bool = False,
):
    """
    Redact a PDF by detecting labelled fields or page titles with OCR.
    Also extracts key fields (DoB, Gender, Latest Appt, File No.) and saves to JSON.

    Args:
        in_pdf: Input PDF path
        out_pdf: Output PDF path
        dpi: Rasterization DPI
        fill: Mask color
        ocr_pipeline: PaddleOCR pipeline
        title_keywords: Dict mapping title names to detection keywords
        title_masks: Dict mapping title names to mask boxes
        search_region: Region to search for title keywords (None = top 30%)
        default_masks: Masks to apply when no title is detected (None = no masks)
        detect_fields_flag: Whether to detect labelled fields instead of using title_masks
        extract_fields_flag: Whether to extract key fields and save to JSON
        verbose: Print detection info
    """
    ensure_dir(out_pdf.parent)
    doc = pymupdf.open(str(in_pdf))
    images: list[Image.Image] = []

    detection_log = []
    has_masks = False
    extracted_fields = {
        "DoB": None,
        "Gender": None,
        "Latest Appt": None,
        "File No.": None,
    }

    json_path = out_pdf.with_suffix(".json")
    if out_pdf.exists() and (not extract_fields_flag or json_path.exists()):
        return True

    try:
        total_pages = doc.page_count

        for i in range(total_pages):
            page = doc.load_page(i)
            img = rasterize_page(page, dpi)

            # Extract key fields from first page only
            if extract_fields_flag and i == 0:
                extracted_fields = extract_key_fields(img, ocr_pipeline)
                extracted_names = [
                    name for name, value in extracted_fields.items() if value is not None
                ]
                detection_log.append(
                    f"Page {i + 1}: Extracted fields: {', '.join(extracted_names) or 'none'}"
                )

            masks = []

            if detect_fields_flag:
                # Try to detect and mask sensitive fields (name/phone/staff/student)
                # Pass extracted_fields only on first page (where fields are extracted)
                fields_to_pass = extracted_fields if (extract_fields_flag and i == 0) else None
                name_results = detect_sensitive_fields(
                    img,
                    ocr_pipeline,
                    extracted_fields=fields_to_pass,
                    field_keywords=field_keywords,
                )
                if name_results:
                    masks = [box for _, box in name_results]
                    detection_log.append(
                        f"Page {i + 1}: Detected {len(name_results)} field(s), applying mask(s)"
                    )
                else:
                    detection_log.append(f"Page {i + 1}: No requested fields detected")
            else:
                # Original logic: detect page title and apply corresponding masks
                detected_title = detect_page_title(img, ocr_pipeline, title_keywords, search_region)

                if verbose:
                    print(f"page {i + 1}: detected title = {detected_title}")

                # Get masks for this page
                if detected_title and detected_title != "flag" and detected_title in title_masks:
                    masks = title_masks[detected_title]
                    detection_log.append(
                        f"Page {i + 1}: Detected '{detected_title}', applying {len(masks)} mask(s)"
                    )
                elif default_masks:
                    masks = default_masks
                    detection_log.append(
                        f"Page {i + 1}: No title detected, applying default {len(masks)} mask(s)"
                    )
                else:
                    detection_log.append(f"Page {i + 1}: No title detected, no masks applied")

            # Apply masks
            if masks:
                img = apply_masks(img, masks, fill)
                has_masks = True

            images.append(img)

        if has_masks:
            images[0].save(str(out_pdf), save_all=True, append_images=images[1:])

        # Save extracted fields to JSON
        if extract_fields_flag:
            json_path = out_pdf.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(extracted_fields, f, ensure_ascii=False, indent=2)
            detection_log.append(f"Extracted fields saved to: {json_path}")

        if verbose:
            print(f"\n{in_pdf.name}:")
            for log in detection_log:
                print(f"  {log}")

    finally:
        doc.close()
        for image in images:
            image.close()

    return has_masks


# =========================
# Worker function for threading
# =========================


def redact_single_pdf(
    pdf: Path,
    input_dir: Path,
    output_dir: Path,
    suffix: str,
    dpi: int,
    fill: str,
    title_keywords: dict[str, list[str]],
    title_masks: dict[str, list[Box]],
    search_region: Box | None,
    default_masks: list[Box] | None,
    field_keywords: list[str] | None = None,
    detect_fields_flag: bool = True,
    extract_fields_flag: bool = False,
    verbose: bool = False,
) -> tuple[Path, bool, str | None]:
    """
    Process a single PDF and return (pdf_path, has_masks, error_msg).
    Each thread gets its own OCR pipeline instance for thread safety.
    """
    try:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR dependencies are missing; install with: uv sync --extra ocr"
            ) from exc

        # PaddleOCR is not thread-safe, so each worker owns a pipeline instance.
        ocr_pipeline = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="cpu",
        )

        rel = pdf.relative_to(input_dir)
        out_pdf = output_dir / rel.parent / f"{pdf.stem}{suffix}.pdf"

        mask_result = redact_pdf_with_ocr(
            in_pdf=pdf,
            out_pdf=out_pdf,
            dpi=dpi,
            fill=fill,
            ocr_pipeline=ocr_pipeline,
            title_keywords=title_keywords,
            title_masks=title_masks,
            search_region=search_region,
            default_masks=default_masks,
            field_keywords=field_keywords,
            detect_fields_flag=detect_fields_flag,
            extract_fields_flag=extract_fields_flag,
            verbose=verbose,
        )

        return (pdf, mask_result, None)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (pdf, False, repr(e))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="emr-extractor redact-detected-fields",
        description="Use OCR to detect and redact labelled fields in PDFs.",
    )

    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--fill", type=str, default="black")
    ap.add_argument("--suffix", type=str, default="_redacted")
    ap.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of worker threads for parallel processing",
    )
    ap.add_argument("--model-home", "--weights-dir", type=Path, default=Path("models"))
    ap.add_argument("--offline", action="store_true", help="use local models without network")
    ap.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELD_KEYWORDS),
        help="Comma-separated field labels to redact",
    )
    ap.add_argument(
        "--detect-fields",
        "--detect-name-field",
        dest="detect_fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect labelled fields instead of using title-based masks",
    )
    ap.add_argument(
        "--extract-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save DoB, Gender, Latest Appt, and File No. to JSON",
    )

    ap.add_argument(
        "--title-keywords",
        type=str,
        help="JSON mapping title names to keyword lists",
        default=None,
    )
    ap.add_argument(
        "--title-masks",
        type=str,
        help="JSON mapping title names to normalized mask boxes",
        default=None,
    )
    ap.add_argument(
        "--search-region",
        type=str,
        default=None,
        help="Normalized title search region; default: [0,0,1,0.25]",
    )
    ap.add_argument(
        "--default-masks",
        type=str,
        default=None,
        help="Default masks when no title detected. Example: [[0.5,0,1,0.2]]",
    )
    ap.add_argument("--verbose", action="store_true", help="Print detection info for each page")

    args = ap.parse_args(argv)
    configure_model_environment("paddleocr", args.model_home, offline=args.offline)

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing; install with: uv sync --extra ocr"
        ) from exc

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")

    ensure_dir(args.output_dir)

    # Parse configurations
    title_keywords = json.loads(args.title_keywords) if args.title_keywords else {}
    if not isinstance(title_keywords, dict):
        raise ValueError("--title-keywords must be a JSON object")

    title_masks_raw = json.loads(args.title_masks) if args.title_masks else {}
    if not isinstance(title_masks_raw, dict):
        raise ValueError("--title-masks must be a JSON object")

    title_masks = {
        k: [norm_box_ratio(*map(float, b)) for b in boxes] for k, boxes in title_masks_raw.items()
    }

    search_region = None
    if args.search_region:
        region_raw = json.loads(args.search_region)
        search_region = norm_box_ratio(*map(float, region_raw))

    default_masks = None
    if args.default_masks:
        masks_raw = json.loads(args.default_masks)
        default_masks = [norm_box_ratio(*map(float, b)) for b in masks_raw]

    field_keywords = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not field_keywords:
        raise ValueError("--fields must contain at least one field label")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be greater than zero")

    # Initialize OCR pipeline is now done per thread for thread safety
    print(f"Starting batch processing with {args.num_threads} threads...")

    # Process PDFs
    pdfs = list(iter_pdfs(args.input_dir, args.recursive))

    if not pdfs:
        raise SystemExit("No PDFs found")

    errors = []
    redacted = []
    unredacted = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                redact_single_pdf,
                pdf,
                args.input_dir,
                args.output_dir,
                args.suffix,
                args.dpi,
                args.fill,
                title_keywords,
                title_masks,
                search_region,
                default_masks,
                field_keywords,
                args.detect_fields,
                args.extract_fields,
                args.verbose,
            ): pdf
            for pdf in pdfs
        }

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Redacting PDFs"):
            pdf_path, mask_result, error = future.result()

            if error:
                errors.append((str(pdf_path), error))
            elif mask_result:
                redacted.append(str(pdf_path.relative_to(args.input_dir)))
            else:
                unredacted.append(str(pdf_path.relative_to(args.input_dir)))

    print(f"\n✅ Redacted files: {len(redacted)}")
    print(f"ℹ️  Files with no detected field: {len(unredacted)}")
    print(f"❌ Failed files: {len(errors)}")

    if redacted:
        log_path = args.output_dir / "redacted_files.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for fpath in redacted:
                f.write(f"{fpath}\n")
        print(f"✅ Redacted files log saved to: {log_path}")

    if unredacted:
        log_path = args.output_dir / "unredacted_files.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for fpath in unredacted:
                f.write(f"{fpath}\n")
        print(f"ℹ️  Unredacted files log saved to: {log_path}")

    if errors:
        print("\n[!] Failed files:")
        for f, e in errors:
            print(f" - {f}\n   {e}")
        raise SystemExit(f"\nDone with errors: {len(errors)}/{len(pdfs)} failed.")
    else:
        print(f"\n✅ Done. Output saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
