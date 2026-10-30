"""Optional small-batch comparison: all originals, selected originals, or originals plus hints."""

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from agents import Agent, RunConfig, Runner, set_default_openai_api

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--with-hints", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-sol")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Configure OPENAI_API_KEY privately")
    if args.with_hints and not args.audit:
        parser.error("--with-hints requires --audit")
    folder = args.folder.resolve()
    records = None
    if args.audit:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        summary = audit.get("summary")
        if summary is None:
            parser.error("The audit contains no completed screening")
        if summary["processing_errors"]:
            parser.error("Resolve screening errors before comparing the same image set")
        records = {r["image_id"]: r for r in summary["selected"]}
        paths = [(folder / image_id).resolve() for image_id in records]
    else:
        paths = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not paths:
        parser.error("No images to review in this condition")
    if any(not p.is_relative_to(folder) or not p.is_file() for p in paths):
        parser.error("An image is missing or outside the chosen folder")
    content = [{"type": "input_text", "text": "Review these fabric images."}]
    for path in paths:
        image_id = path.relative_to(folder).as_posix()
        text = f"image_id: {image_id}"
        if args.with_hints:
            text += "\nDetector predictions (unverified): " + json.dumps(records[image_id])
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.extend([
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "high"},
        ])
    args.output.mkdir(parents=True, exist_ok=False)
    set_default_openai_api("responses")
    agent = Agent(
        name="Fabric comparison reviewer", model=args.model,
        instructions=("Inspect each fabric image for broken picks, holes, missing picks, "
                      "and thick weft. Report image_id, location, visible evidence, "
                      "uncertainty, and a next step for human review. Treat any supplied "
                      "detector hints as predictions. Do not infer manufacturing causes "
                      "or physical defect sizes. Do not invent detector scores."),
    )
    started = time.perf_counter()
    result = Runner.run_sync(agent, [{"role": "user", "content": content}],
                             run_config=RunConfig(tracing_disabled=True))
    usage = result.context_wrapper.usage
    metrics = {"model": args.model, "images_submitted": len(paths),
               "condition": "all_originals" if records is None else
                            ("selected_originals_with_hints" if args.with_hints else "selected_originals"),
               "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
               "total_tokens": usage.total_tokens,
               "elapsed_seconds": time.perf_counter() - started}
    (args.output / "report.md").write_text(str(result.final_output), encoding="utf-8")
    (args.output / "usage.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
