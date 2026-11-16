"""Run a Roboflow Gemini Open Prompt Workflow and draw validated screw boxes.

The parser depends on a documented JSON schema, not on a Gemini model name.
The Workflow needs two outputs: boxes_json and prompt_echo.
"""

import json
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

# Configure the Gemini block's prompt as $inputs.prompt (not hardcoded text).
# Use these same instructions for the editor's prompt input during testing.
PROMPT = """Detect visible screws at the upper and lower mounting positions on the
product's central structure. Use one box for each full visible fastener
region, including its outer exposed end and inner metal feature.
Exclude empty positions, plastic ribs, circular plastic features,
and reflections. Do not add a screw because it is expected there.

Return only a JSON list without Markdown fences or extra explanation.
Use the key "label" with the value "screw" for every object.
Use the key "box_2d" with [ymin, xmin, ymax, xmax] coordinates.
Use integers normalized to 0–1000 relative to the full image.
Return [] if no screw is visible."""

MAX_TEXT = 1_000_000
MAX_DEPTH = 15
MAX_BOXES = 1000
WRAPPER_KEYS = (
    "boxes_json", "boxes", "detections", "objects", "predictions",
    "output", "output_text", "response", "result", "results",
    "data", "text", "message", "content", "candidates", "parts",
)


def _parse_text(text):
    """Accept JSON, one JSON fence, or prose surrounding one JSON value."""
    source = text.strip().lstrip("\ufeff").strip()
    if not source or len(source) > MAX_TEXT:
        raise ValueError("Gemini response is empty or too long")
    try:
        return json.loads(source)
    except json.JSONDecodeError:
        if source.startswith(("[", "{")):
            raise ValueError("Malformed JSON or multiple JSON values")

    fenced = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", source, flags=re.I))
    if fenced:
        if len(fenced) != 1:
            raise ValueError("Expected exactly one JSON code block")
        block = fenced[0]
        outside = source[:block.start()] + source[block.end():]
        if any(char in outside for char in "[]{}`"):
            raise ValueError("Ambiguous content outside the JSON code block")
        return json.loads(block.group(1).strip())

    # Parse from the first opening bracket. Never salvage an inner object
    # from a malformed outer array or silently choose among multiple results.
    openings = [p for p in (source.find("["), source.find("{")) if p >= 0]
    if not openings:
        raise ValueError("Gemini did not return a JSON array or object")
    position = min(openings)
    value, consumed = json.JSONDecoder().raw_decode(source[position:])
    if any(char in source[position + consumed:] for char in "[]{}"):
        raise ValueError("Multiple or malformed JSON values")
    return value


def _check_coordinate_metadata(item):
    """Reject declared pixel, 0-1, or XYXY coordinates rather than misplot."""
    for key in ("coordinate_system", "coordinate_units", "units", "coordinate_format", "box_format", "coordinate_order"):
        if key not in item:
            continue
        hint = str(item[key]).strip().lower().replace(" ", "")
        scale_01 = bool(re.search(r"(?<!\d)0(?:-|\.\.)1(?!\d)", hint))
        if ("pixel" in hint or "xyxy" in hint or "xmin,ymin" in hint
                or scale_01 or "[0,1]" in hint):
            raise ValueError(f"Unsupported coordinate format specified in {key}: {item[key]!r}")


def _unwrap(value, depth=0):
    """Remove common Roboflow, Gemini, and Open Prompt response wrappers."""
    if depth >= MAX_DEPTH:
        raise ValueError("Gemini response has too many nested wrappers")

    if isinstance(value, str):
        return _unwrap(_parse_text(value), depth + 1)

    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, dict) and (
            "box_2d" in item or "bbox_2d" in item or "bounding_box" in item
        ) for item in value):
            return value
        if len(value) == 1:
            return _unwrap(value[0], depth + 1)
        raise ValueError("Expected detections or one Workflow response, not multiple responses")

    if isinstance(value, dict):
        _check_coordinate_metadata(value)
        if value.get("error_status") is True:
            raise ValueError("The Gemini block reported an output-decoding error")
        if any(k in value for k in ("box_2d", "bbox_2d", "bounding_box")):
            return [value]
        if "parts" in value and isinstance(value["parts"], list):
            parts = value["parts"]
            if all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts):
                return _unwrap("".join(p["text"] for p in parts), depth + 1)
        keys = [key for key in WRAPPER_KEYS if key in value]
        if len(keys) > 1:
            raise ValueError("Multiple output fields; cannot choose a result safely")
        if keys:
            return _unwrap(value[keys[0]], depth + 1)
        raise ValueError("No boxes_json, boxes, or recognizable Gemini output field")

    raise ValueError(f"Unsupported Gemini response type: {type(value).__name__}")


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    # Gemini occasionally returns numeric JSON strings. Accept only simple
    # numeric strings, not text, units, or expressions.
    if isinstance(value, str) and re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", value):
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if not 0 <= value <= 1000:
        raise ValueError(f"{label} must be normalized between 0 and 1000")
    return value


def _coordinates(item, index):
    keys = [key for key in ("box_2d", "bbox_2d", "bounding_box") if key in item]
    if len(keys) != 1:
        raise ValueError(f"Detection {index} needs exactly one coordinate field")
    coords = item[keys[0]]
    if keys[0] != "box_2d" and not isinstance(coords, dict):
        raise ValueError("Alternate box fields must use explicitly named coordinates")
    if isinstance(coords, dict):
        # Named coordinates make the order unambiguous.
        aliases = (
            ("ymin", "xmin", "ymax", "xmax"),
            ("y_min", "x_min", "y_max", "x_max"),
        )
        for names in aliases:
            if all(name in coords for name in names):
                coords = [coords[name] for name in names]
                break
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise ValueError(
            f"Detection {index} needs box_2d [ymin, xmin, ymax, xmax] "
            "or a bounding_box object with explicitly named coordinates"
        )
    vals = [_number(v, f"Detection {index} coordinate {i}") for i, v in enumerate(coords, 1)]
    ymin, xmin, ymax, xmax = vals
    if ymin >= ymax or xmin >= xmax:
        raise ValueError(f"Detection {index} has a reversed or empty box")
    return vals


def decode_boxes(response):
    """Normalize valid Gemini box JSON to [{label, box_2d}].

    Assumes the prompt's [ymin, xmin, ymax, xmax] ordering and 0..1000 units.
    Unrecognized schemas fail clearly rather than drawing incorrect boxes.
    """
    boxes = _unwrap(response)
    if len(boxes) > MAX_BOXES:
        raise ValueError("Unexpectedly large number of detections")
    normalized = []
    for index, item in enumerate(boxes, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Detection {index} is not a JSON object")
        label = next((item.get(k) for k in ("label", "class", "class_name") if k in item), None)
        if not isinstance(label, str) or label.strip().casefold() != "screw":
            raise ValueError(f"Detection {index} has missing or unexpected label {label!r}")
        _check_coordinate_metadata(item)
        normalized.append({"label": "screw", "box_2d": _coordinates(item, index)})
    return normalized


def visualize(image, boxes, output_path):
    image = image.copy()
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for item in boxes:
        ymin, xmin, ymax, xmax = item["box_2d"]
        left = min(width - 1, max(0, math.floor(xmin * width / 1000)))
        right = min(width - 1, max(left, math.ceil(xmax * width / 1000) - 1))
        top = min(height - 1, max(0, math.floor(ymin * height / 1000)))
        bottom = min(height - 1, max(top, math.ceil(ymax * height / 1000) - 1))
        draw.rectangle((left, top, right, bottom), outline="#6706CE", width=3)
        caption_top = max(0, top - 18)
        draw.rectangle((left, caption_top, left + 45, caption_top + 17), fill="#6706CE")
        draw.text((left + 3, caption_top + 2), "screw", fill="white")
    image.save(output_path)


def verify_workflow_response(results, prompt):
    """Check the same text reached the Workflow's named prompt input."""
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Expected one Roboflow Workflow result for one image")
    result = results[0]
    if "prompt_echo" not in result:
        raise ValueError("Add a Workflow output named prompt_echo selecting $inputs.prompt")
    if result["prompt_echo"] != prompt:
        raise ValueError("Prompt mismatch. Check the Workflow's prompt input and its output binding")
    if "boxes_json" not in result:
        raise ValueError("Add a Workflow output named boxes_json selecting $steps.gemini_boxes.output")
    return result["boxes_json"]


def main():
    import numpy as np
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    if not PROMPT.strip():
        raise ValueError("PROMPT must not be empty")
    image = ImageOps.exif_transpose(Image.open("inspection.jpg")).convert("RGB")
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    ).configure(InferenceConfiguration(api_key_transport="header"))

    results = client.run_workflow(
        workspace_name="YOUR_WORKSPACE",
        workflow_id="YOUR_WORKFLOW_ID",
        images={"image": np.asarray(image)[:, :, ::-1].copy()},
        parameters={"prompt": PROMPT},
        use_cache=False,
    )

    # Save the complete response and supplied prompt even if parsing fails.
    Path("gemini_workflow_response.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    Path("gemini_prompt.txt").write_text(PROMPT, encoding="utf-8")
    raw = verify_workflow_response(results, PROMPT)
    Path("gemini_boxes.txt").write_text(
        raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    boxes = decode_boxes(raw)
    Path("gemini_boxes_normalized.json").write_text(
        json.dumps(boxes, indent=2), encoding="utf-8"
    )
    visualize(image, boxes, "gemini_boxes_preview.png")
    print(f"Saved gemini_boxes_preview.png with {len(boxes)} boxes for review")
    if not boxes:
        print("Gemini returned no boxes. This does not prove that screws are absent.")


if __name__ == "__main__":
    main()
