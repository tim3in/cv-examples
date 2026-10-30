"""Use a published Roboflow Workflow as a tool for a Python inspection agent."""

import argparse
import base64
import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from agents import (
    Agent, ModelSettings, RunConfig, Runner, ToolOutputImage, ToolOutputText,
    function_tool, set_default_openai_api,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from PIL import Image

WORKSPACE = "tim-4ijf0"
WORKFLOW_ID = "textile-defect-agent-filter-1793090127138"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
DEFECT_CLASSES = {
    "fabric_broken_pick", "fabric_hole",
    "fabric_missing_pick", "fabric_thick_weft",
}
CROP_MAX_SIDE = 1024


def validate_result(response, image_id):
    """Reject inconsistent screening data before offering it to the agent."""
    if not isinstance(response, list) or len(response) != 1:
        raise ValueError("Expected one Workflow result")
    result = response[0]["result"]
    if (result["processing_status"] != "success"
            or result["image_id"] != image_id):
        raise ValueError("Unsuccessful or mismatched result")
    predictions, count = result["predictions"], result["defect_count"]
    if (not isinstance(predictions, list) or type(count) is not int
            or count != len(predictions)):
        raise ValueError("Invalid prediction count")
    selected = result["selected_for_review"]
    if type(selected) is not bool or selected != (count > 0):
        raise ValueError("Invalid review decision")
    if result["reason"] != ("possible_defect" if selected else "not_selected"):
        raise ValueError("Invalid selection reason")
    dims = result["image_dimensions"]
    if any(type(dims[k]) is not int or dims[k] <= 0 for k in ("width", "height")):
        raise ValueError("Invalid image dimensions")
    expected = {"format": "xyxy", "units": "pixels", "reference": "original_image",
                "origin": "top_left", "x_direction": "right", "y_direction": "down"}
    if any(result["coordinate_format"].get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected coordinate format")
    for hint in predictions:
        score = float(hint["confidence"])
        box = hint["bbox"]
        x1, y1, x2, y2 = [float(box[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
        if (hint["class"] not in DEFECT_CLASSES or not math.isfinite(score)
                or not 0 <= score <= 1
                or not all(math.isfinite(v) for v in (x1, y1, x2, y2))
                or not x1 < x2 or not y1 < y2
                or x2 <= 0 or y2 <= 0 or x1 >= dims["width"] or y1 >= dims["height"]):
            raise ValueError("Invalid defect hint")
    return result


def build_tools(folder, client, confidence=0.25,
                workspace=WORKSPACE, workflow_id=WORKFLOW_ID):
    """Keep each run's files and results in its own local state."""
    folder = Path(folder).resolve()
    if not folder.is_dir() or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Provide an existing folder and confidence from 0 to 1")
    paths = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not paths:
        raise ValueError("No supported image files found")
    selected_files = {}
    state = {"summary": None, "records": [], "views": [], "view_errors": []}

    @function_tool
    def screen_fabric_images() -> str:
        """Screen the configured folder with Roboflow before viewing any image.

        Returns JSON counts, selected-image hints and processing errors.
        No image pixels are returned. Repeated calls reuse this run's results.
        """
        if state["summary"] is not None:
            return json.dumps(state["summary"])
        print("[tool] screen_fabric_images")
        selected, errors = [], []
        for path in paths:
            image_id = path.relative_to(folder).as_posix()
            try:
                response = client.run_workflow(
                    workspace_name=workspace, workflow_id=workflow_id,
                    images={"image": str(path)},
                    parameters={"image_id": image_id,
                                "confidence_threshold": confidence},
                    use_cache=False,
                )
                result = validate_result(response, image_id)
                state["records"].append({"path": str(path), "result": result})
                if result["selected_for_review"]:
                    selected_files[image_id] = (path, result)
                    selected.append({
                        "image_id": image_id,
                        "image_dimensions": result["image_dimensions"],
                        "hints": [{"defect_index": i, **hint}
                                  for i, hint in enumerate(result["predictions"])],
                    })
            except Exception as exc:
                errors.append({"image_id": image_id, "error_type": type(exc).__name__})
        state["summary"] = {
            "images_total": len(paths), "screened": len(state["records"]),
            "selected_count": len(selected),
            "not_selected": len(state["records"]) - len(selected),
            "selected": selected, "processing_errors": errors,
        }
        return json.dumps(state["summary"])

    @function_tool
    def view_defect(image_id: str, defect_index: int) -> list[ToolOutputText | ToolOutputImage]:
        """Show one padded region from an image selected by screen_fabric_images.

        Args:
            image_id: An identifier in the screening tool's selected list.
            defect_index: The zero-based hint index returned by that tool.
        """
        print(f"[tool] view_defect: {image_id}, hint {defect_index}")
        try:
            path, result = selected_files[image_id]
            data_url, bounds, size = make_crop(path, result, defect_index)
            record = {"image_id": image_id, "defect_index": defect_index,
                      "crop_xyxy": list(bounds), "display_size": list(size)}
            state["views"].append(record)
            return [ToolOutputText(text=json.dumps(record)),
                    ToolOutputImage(image_url=data_url, detail="high")]
        except Exception as exc:
            record = {"image_id": image_id, "defect_index": defect_index,
                      "error_type": type(exc).__name__}
            state["view_errors"].append(record)
            return [ToolOutputText(text=json.dumps({"view_error": record}))]

    return [screen_fabric_images, view_defect], state


def make_crop(path, result, defect_index):
    """Read named xyxy coordinates and return a bounded crop as a data URL."""
    if not 0 <= defect_index < len(result["predictions"]):
        raise ValueError("Unknown defect index")
    box = result["predictions"][defect_index]["bbox"]
    x1, y1, x2, y2 = [float(box[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
    with Image.open(path) as source:
        image = source.convert("RGB")
    dims = result["image_dimensions"]
    if image.size != (dims["width"], dims["height"]):
        raise ValueError("Original image dimensions changed")
    pad_x = 0.25 * (x2 - x1) + 32
    pad_y = 0.25 * (y2 - y1) + 32
    bounds = (max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
              min(image.width, math.ceil(x2 + pad_x)),
              min(image.height, math.ceil(y2 + pad_y)))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("Empty crop")
    crop = image.crop(bounds)
    crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=95)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}", bounds, crop.size


def create_run_directory(base_dir: Path) -> Path:
    """Create a unique run folder using the computer's local date and time."""
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    suffix = 0
    while True:
        name = timestamp if suffix == 0 else f"{timestamp}_{suffix}"
        run_dir = base_dir / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            suffix += 1


INSTRUCTIONS = """You review fabric inspection images using two tools.
Call screen_fabric_images first. It screens the configured folder.
Treat all returned labels, scores, and boxes as predictions.
For every returned hint, call view_defect with its image_id and defect_index.
Use the image returned by the tool to assess that region. A high detection
score alone does not establish a defect. Do not claim to see unavailable images.
Classify each reviewed hint as visually supported, not supported, or uncertain.
Report a failed or missing crop as not reviewed. Crops show only local regions;
do not make claims about the full image or about images that were not selected.
Keep processing errors separate from successful not_selected results.
Write a short report with one entry per hint: image_id, defect_index, predicted
class, detector score, original bbox, visual evidence, and review assessment.
Finish with image counts, region-review counts, errors, and next steps for a
human inspector. Do not infer manufacturing causes or physical defect sizes.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("fabric-agent-results"),
                        help="Parent folder for timestamped run directories")
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--workflow", default=WORKFLOW_ID)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()
    for key in ("ROBOFLOW_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            parser.error(f"Configure {key} privately in this terminal")
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    ).configure(InferenceConfiguration(api_key_transport="header"))
    tools, state = build_tools(args.folder, client, args.confidence,
                               args.workspace, args.workflow)
    output_dir = create_run_directory(args.output)
    print(f"[output] {output_dir}")
    set_default_openai_api("responses")
    agent = Agent(name="Fabric inspection agent", model=args.model,
                  instructions=INSTRUCTIONS, tools=tools,
                  model_settings=ModelSettings(parallel_tool_calls=False))
    started = time.perf_counter()
    try:
        result = Runner.run_sync(
            agent, "Inspect the configured fabric folder and write the review report.",
            max_turns=args.max_turns, run_config=RunConfig(tracing_disabled=True),
        )
    except Exception as exc:
        (output_dir / "partial-run.json").write_text(
            json.dumps({"run_error_type": type(exc).__name__, "audit": state}, indent=2),
            encoding="utf-8",
        )
        print(f"Run incomplete ({type(exc).__name__}); "
              f"inspect {output_dir / 'partial-run.json'}.")
        return 2
    usage = result.context_wrapper.usage
    requested = {(item["image_id"], hint["defect_index"])
                 for item in (state["summary"] or {}).get("selected", [])
                 for hint in item["hints"]}
    returned = {(v["image_id"], v["defect_index"]) for v in state["views"]}
    metrics = {
        "model": args.model, "workflow": args.workflow, "confidence": args.confidence,
        "elapsed_seconds": time.perf_counter() - started,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "screening_called": state["summary"] is not None,
        "regions_requested": len(requested), "regions_returned": len(returned),
        "regions_without_image": [list(pair) for pair in sorted(requested - returned)],
    }
    summary = state["summary"] or {}
    for key in ("images_total", "screened", "selected_count", "not_selected"):
        metrics[key] = summary.get(key, 0)
    metrics["processing_errors"] = len(summary.get("processing_errors", []))
    metrics["view_errors"] = len(state["view_errors"])
    (output_dir / "report.md").write_text(str(result.final_output), encoding="utf-8")
    for name, data in (("audit.json", state), ("usage.json", metrics)):
        (output_dir / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(result.final_output)
    print(json.dumps(metrics, indent=2))
    print(f"Results saved to: {output_dir}")
    complete = (metrics["screening_called"] and not metrics["regions_without_image"]
                and not metrics["processing_errors"] and not metrics["view_errors"])
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
