"""
Process a pre-recorded conveyor video with a Roboflow Workflow.

This version:
- Uses cumulative `count_out` as the completed-product counter.
- Calculates per-unit conveyor cycle time.
- Uses the Supervision Roboflow color palette for a compact video overlay.
- Displays every metric on a separate line in smaller text.
- Writes an annotated output video.
- Saves an evidence image for each completed product.
- Continuously updates a JSON event log.

Install:
    pip install "inference-sdk[webrtc]" opencv-python numpy supervision

Set the API key before running:
    Windows PowerShell:
        $env:ROBOFLOW_API_KEY="YOUR_NEW_KEY"

    macOS/Linux:
        export ROBOFLOW_API_KEY="YOUR_NEW_KEY"

Run:
    python roboflow_video_cycle_logger_supervision.py
"""

from __future__ import annotations

import base64
import json
import os
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoFileSource, VideoMetadata


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

INPUT_VIDEO = Path("package_processing.mp4")

OUTPUT_DIRECTORY = Path("shop_floor_cycle_data")
OUTPUT_VIDEO = OUTPUT_DIRECTORY / "output_annotated.mp4"
JSON_OUTPUT = OUTPUT_DIRECTORY / "product_cycle_events.json"
EVIDENCE_DIR = OUTPUT_DIRECTORY / "evidence"

API_URL = "https://serverless.roboflow.com"
WORKSPACE = "tim-4ijf0"
WORKFLOW = "product-cycle-time"
IMAGE_INPUT = "image"

VIDEO_OUTPUT = "output_image"
COUNT_OUT_OUTPUT = "count_out"

REQUESTED_PLAN = "webrtc-gpu-medium"
REQUESTED_REGION = "us"
PROCESSING_TIMEOUT_SECONDS = 3600

# A cycle longer than this is marked as slow.
SLOW_CYCLE_SECONDS = 6.0

# Preview settings.
DISPLAY_PREVIEW = True

# False processes and displays results as quickly as they arrive.
# True adds a source-FPS delay to the local preview.
PREVIEW_AT_SOURCE_SPEED = False
PREVIEW_WINDOW_NAME = "Roboflow Product Cycle Time"

# Update JSON periodically even when no new unit crosses the line.
JSON_PROGRESS_EVERY_N_FRAMES = 30

# Evidence images are taken from the Workflow-annotated frame before
# the local metric panel is added.
SAVE_EVIDENCE_IMAGES = True

# Production metadata written into every event.
FACILITY_ID = "plant-jaipur"
LINE_ID = "line-2"
STATION_ID = "conveyor-exit"
CAMERA_ID = "camera-07"
SHIFT = "A"
SKU = "CTN-500-B"
WORK_ORDER = "WO-5187"
PRODUCT_NAME = "sealed_carton"


# ---------------------------------------------------------------------------
# SUPERVISION / ROBOFLOW UI COLORS
# ---------------------------------------------------------------------------

ROBOFLOW_PALETTE = sv.ColorPalette.ROBOFLOW

PURPLE = ROBOFLOW_PALETTE.by_idx(1).as_bgr()
LAVENDER = ROBOFLOW_PALETTE.by_idx(0).as_bgr()
BLUE = ROBOFLOW_PALETTE.by_idx(3).as_bgr()
CYAN = ROBOFLOW_PALETTE.by_idx(4).as_bgr()
GREEN = ROBOFLOW_PALETTE.by_idx(5).as_bgr()
ORANGE = ROBOFLOW_PALETTE.by_idx(7).as_bgr()
PINK = ROBOFLOW_PALETTE.by_idx(8).as_bgr()

WHITE = sv.Color.WHITE.as_bgr()
LIGHT_TEXT = (225, 225, 235)
MUTED_TEXT = (175, 175, 190)
PANEL_BACKGROUND = (22, 18, 30)
DIVIDER_COLOR = (70, 60, 85)


# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def format_video_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--:--.---"

    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_seconds(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.3f} s"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so a reader never sees an incomplete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def unwrap_workflow_value(value: Any) -> Any:
    """
    Workflow outputs may be returned directly or wrapped in dictionaries such
    as {"value": ...}. Repeatedly unwrap common wrappers.
    """
    current = value

    for _ in range(5):
        if isinstance(current, dict) and "value" in current:
            current = current["value"]
            continue
        break

    return current


def extract_counter(value: Any, field_name: str) -> Optional[int]:
    """Convert the count_out Workflow output into an integer."""
    value = unwrap_workflow_value(value)

    if value is None:
        return None

    if isinstance(value, dict):
        if field_name in value:
            return extract_counter(value[field_name], field_name)

        if "count" in value:
            return extract_counter(value["count"], field_name)

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return extract_counter(value[0], field_name)

    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def decode_workflow_image(value: Any) -> Optional[np.ndarray]:
    """Decode a base64 Workflow image output into an OpenCV BGR frame."""
    encoded = unwrap_workflow_value(value)

    if encoded is None:
        return None

    if isinstance(encoded, bytes):
        raw_bytes = encoded

    elif isinstance(encoded, str):
        # Support a standard data URI as well as plain base64.
        if "," in encoded and encoded.lstrip().startswith("data:"):
            encoded = encoded.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(encoded)
        except (ValueError, TypeError):
            return None

    else:
        return None

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)

    if image_array.size == 0:
        return None

    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def metadata_video_seconds(
    metadata: VideoMetadata,
    fallback_fps: float,
) -> tuple[float, str]:
    """
    Calculate the frame time within the source video.

    Prefer presentation timestamp × time base. If unavailable, use frame ID
    divided by the source FPS.
    """
    pts = getattr(metadata, "pts", None)
    time_base = getattr(metadata, "time_base", None)

    if pts is not None and time_base is not None:
        try:
            return float(pts * time_base), "video_pts"
        except (TypeError, ValueError, OverflowError):
            try:
                return float(pts) * float(time_base), "video_pts"
            except (TypeError, ValueError, OverflowError):
                pass

    frame_id = int(getattr(metadata, "frame_id", 0) or 0)
    safe_fps = fallback_fps if fallback_fps > 0 else 30.0

    return frame_id / safe_fps, "frame_id_over_fps"


def inspect_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0

    duration_seconds = frame_count / fps if frame_count > 0 else None

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


# ---------------------------------------------------------------------------
# COMPACT SUPERVISION-STYLE METRIC PANEL
# ---------------------------------------------------------------------------

def draw_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    thickness: int = -1,
) -> None:
    """Draw a rounded rectangle using OpenCV primitives."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)

    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

        cv2.ellipse(
            image,
            (x1 + radius, y1 + radius),
            (radius, radius),
            180,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            image,
            (x2 - radius, y1 + radius),
            (radius, radius),
            270,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            image,
            (x2 - radius, y2 - radius),
            (radius, radius),
            0,
            0,
            90,
            color,
            thickness,
        )
        cv2.ellipse(
            image,
            (x1 + radius, y2 - radius),
            (radius, radius),
            90,
            0,
            90,
            color,
            thickness,
        )


def put_small_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def status_display_color(status: str) -> tuple[int, int, int]:
    if status == "normal":
        return GREEN

    if status == "slow":
        return ORANGE

    if status in {"counter_reset", "cycle_unavailable"}:
        return PINK

    return LAVENDER


def add_metrics_panel(
    frame: np.ndarray,
    *,
    video_seconds: float,
    count_out: int,
    latest_cycle_seconds: Optional[float],
    average_cycle_seconds: Optional[float],
    median_cycle_seconds: Optional[float],
    estimated_units_per_minute: Optional[float],
    slow_cycle_count: int,
    cycle_status: str,
) -> np.ndarray:
    """
    Add a compact Supervision/Roboflow-style metrics panel.

    Every field is displayed on its own line. Labels use muted text while
    values use distinct colors from the Supervision Roboflow palette.
    """
    output = frame.copy()

    height, width = output.shape[:2]

    # Scale the panel and text with resolution, but keep text deliberately small.
    ui_scale = max(0.78, min(1.15, width / 1280.0))
    text_scale = max(0.36, min(0.48, 0.40 * ui_scale))
    heading_scale = text_scale + 0.05
    text_thickness = 1

    margin = max(10, int(14 * ui_scale))
    panel_width = min(
        max(int(320 * ui_scale), 280),
        max(280, width - (2 * margin)),
    )

    header_height = int(34 * ui_scale)
    row_height = int(27 * ui_scale)
    footer_padding = int(10 * ui_scale)

    rows: list[tuple[str, str, tuple[int, int, int]]] = [
        ("Video time", format_video_time(video_seconds), CYAN),
        ("Completed units", str(count_out), LAVENDER),
        ("Latest cycle", format_seconds(latest_cycle_seconds), BLUE),
        ("Average cycle", format_seconds(average_cycle_seconds), PURPLE),
        ("Median cycle", format_seconds(median_cycle_seconds), PINK),
        (
            "Production rate",
            (
                "--"
                if estimated_units_per_minute is None
                else f"{estimated_units_per_minute:.2f} units/min"
            ),
            GREEN,
        ),
        ("Slow cycles", str(slow_cycle_count), ORANGE),
        ("Status", cycle_status.replace("_", " ").title(), status_display_color(cycle_status)),
    ]

    panel_height = (
        header_height
        + (len(rows) * row_height)
        + footer_padding
    )

    x1 = margin
    y1 = margin
    x2 = min(width - margin, x1 + panel_width)
    y2 = min(height - margin, y1 + panel_height)

    # Use a translucent dark panel over the Workflow output.
    panel_overlay = output.copy()

    draw_rounded_rectangle(
        panel_overlay,
        (x1, y1),
        (x2, y2),
        PANEL_BACKGROUND,
        radius=max(8, int(12 * ui_scale)),
        thickness=-1,
    )

    cv2.addWeighted(panel_overlay, 0.84, output, 0.16, 0, output)

    # Thin Roboflow-purple outline.
    draw_rounded_rectangle(
        output,
        (x1, y1),
        (x2, y2),
        PURPLE,
        radius=max(8, int(12 * ui_scale)),
        thickness=1,
    )

    # Colored header accent.
    header_overlay = output.copy()

    draw_rounded_rectangle(
        header_overlay,
        (x1 + 1, y1 + 1),
        (x2 - 1, y1 + header_height),
        PURPLE,
        radius=max(7, int(11 * ui_scale)),
        thickness=-1,
    )

    # Cover the bottom rounded portion so only the top corners remain rounded.
    cv2.rectangle(
        header_overlay,
        (x1 + 1, y1 + header_height - max(6, int(8 * ui_scale))),
        (x2 - 1, y1 + header_height),
        PURPLE,
        -1,
    )

    cv2.addWeighted(header_overlay, 0.82, output, 0.18, 0, output)

    put_small_text(
        output,
        "SHOP FLOOR METRICS",
        (
            x1 + int(12 * ui_scale),
            y1 + int(23 * ui_scale),
        ),
        WHITE,
        heading_scale,
        text_thickness,
    )

    label_x = x1 + int(14 * ui_scale)
    value_x = x1 + int(142 * ui_scale)

    current_y = y1 + header_height

    for index, (label, value, value_color) in enumerate(rows):
        current_y += row_height

        baseline_y = current_y - int(8 * ui_scale)

        # Small palette marker, similar to Supervision label styling.
        cv2.circle(
            output,
            (
                label_x,
                baseline_y - max(1, int(2 * ui_scale)),
            ),
            max(2, int(3 * ui_scale)),
            value_color,
            -1,
            cv2.LINE_AA,
        )

        put_small_text(
            output,
            label,
            (
                label_x + int(10 * ui_scale),
                baseline_y,
            ),
            MUTED_TEXT,
            text_scale,
            text_thickness,
        )

        put_small_text(
            output,
            value,
            (
                value_x,
                baseline_y,
            ),
            value_color,
            text_scale,
            text_thickness,
        )

        if index < len(rows) - 1:
            divider_y = current_y

            cv2.line(
                output,
                (
                    x1 + int(12 * ui_scale),
                    divider_y,
                ),
                (
                    x2 - int(12 * ui_scale),
                    divider_y,
                ),
                DIVIDER_COLOR,
                1,
                cv2.LINE_AA,
            )

    return output


# ---------------------------------------------------------------------------
# APPLICATION STATE AND SUMMARY
# ---------------------------------------------------------------------------

@dataclass
class ProcessingState:
    source_info: dict[str, Any]
    started_at_utc: str = field(default_factory=utc_now_iso)

    previous_count_out: int = 0
    latest_count_out: int = 0

    production_sequence: int = 0
    last_completion_video_seconds: Optional[float] = None
    latest_cycle_seconds: Optional[float] = None
    latest_cycle_status: str = "waiting_for_first_unit"

    cycle_values: list[float] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    processed_frames: int = 0
    latest_frame_id: Optional[int] = None
    latest_video_seconds: Optional[float] = None

    first_event_video_seconds: Optional[float] = None
    last_event_video_seconds: Optional[float] = None

    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[tuple[int, int]] = None

    lock: threading.Lock = field(default_factory=threading.Lock)


def build_summary(state: ProcessingState) -> dict[str, Any]:
    cycles = state.cycle_values

    average = statistics.fmean(cycles) if cycles else None
    median = statistics.median(cycles) if cycles else None
    minimum = min(cycles) if cycles else None
    maximum = max(cycles) if cycles else None

    standard_deviation = (
        statistics.pstdev(cycles)
        if len(cycles) > 1
        else (0.0 if cycles else None)
    )

    units_per_minute = 60.0 / average if average and average > 0 else None
    units_per_hour = 3600.0 / average if average and average > 0 else None

    slow_cycle_count = sum(
        cycle > SLOW_CYCLE_SECONDS
        for cycle in cycles
    )

    slow_cycle_frequency = (
        (slow_cycle_count / len(cycles)) * 100.0
        if cycles
        else 0.0
    )

    production_span_seconds = None

    if (
        state.first_event_video_seconds is not None
        and state.last_event_video_seconds is not None
    ):
        production_span_seconds = max(
            0.0,
            state.last_event_video_seconds - state.first_event_video_seconds,
        )

    return {
        "processed_frames": state.processed_frames,
        "latest_frame_id": state.latest_frame_id,
        "latest_video_seconds": state.latest_video_seconds,
        "latest_video_time": format_video_time(state.latest_video_seconds),

        "count_out": state.latest_count_out,
        "production_counter": "count_out",
        "production_units_logged": state.production_sequence,

        "valid_cycle_measurements": len(cycles),
        "latest_cycle_seconds": (
            round(state.latest_cycle_seconds, 3)
            if state.latest_cycle_seconds is not None
            else None
        ),
        "average_cycle_seconds": (
            round(average, 3)
            if average is not None
            else None
        ),
        "median_cycle_seconds": (
            round(median, 3)
            if median is not None
            else None
        ),
        "minimum_cycle_seconds": (
            round(minimum, 3)
            if minimum is not None
            else None
        ),
        "maximum_cycle_seconds": (
            round(maximum, 3)
            if maximum is not None
            else None
        ),
        "cycle_standard_deviation_seconds": (
            round(standard_deviation, 3)
            if standard_deviation is not None
            else None
        ),

        "estimated_units_per_minute": (
            round(units_per_minute, 3)
            if units_per_minute is not None
            else None
        ),
        "estimated_units_per_hour": (
            round(units_per_hour, 3)
            if units_per_hour is not None
            else None
        ),

        "slow_cycle_threshold_seconds": SLOW_CYCLE_SECONDS,
        "slow_cycle_count": slow_cycle_count,
        "slow_cycle_frequency_percent": round(slow_cycle_frequency, 2),

        "first_event_video_seconds": state.first_event_video_seconds,
        "first_event_video_time": (
            format_video_time(state.first_event_video_seconds)
            if state.first_event_video_seconds is not None
            else None
        ),
        "last_event_video_seconds": state.last_event_video_seconds,
        "last_event_video_time": (
            format_video_time(state.last_event_video_seconds)
            if state.last_event_video_seconds is not None
            else None
        ),
        "production_span_seconds": production_span_seconds,
    }


def build_document(
    state: ProcessingState,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "status": status,
        "updated_at_utc": utc_now_iso(),

        "session": {
            "started_at_utc": state.started_at_utc,
            "source_video": str(INPUT_VIDEO),
            "output_video": str(OUTPUT_VIDEO),
            "evidence_directory": str(EVIDENCE_DIR),

            "workspace": WORKSPACE,
            "workflow": WORKFLOW,
            "api_url": API_URL,

            "requested_plan": REQUESTED_PLAN,
            "requested_region": REQUESTED_REGION,

            "source_video_info": state.source_info,
        },

        "production_context": {
            "facility_id": FACILITY_ID,
            "line_id": LINE_ID,
            "station_id": STATION_ID,
            "camera_id": CAMERA_ID,

            "shift": SHIFT,
            "sku": SKU,
            "work_order": WORK_ORDER,
            "product_name": PRODUCT_NAME,
        },

        "summary": build_summary(state),
        "warnings": state.warnings,
        "events": state.events,
    }


def save_live_document(
    state: ProcessingState,
    status: str = "processing",
) -> None:
    atomic_write_json(
        JSON_OUTPUT,
        build_document(state, status),
    )


def save_evidence_frame(
    frame: Optional[np.ndarray],
    event_id: str,
) -> Optional[str]:
    if not SAVE_EVIDENCE_IMAGES or frame is None:
        return None

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    path = EVIDENCE_DIR / f"{event_id}.jpg"

    if not cv2.imwrite(str(path), frame):
        return None

    return str(path)


def initialize_video_writer(
    state: ProcessingState,
    frame: np.ndarray,
) -> cv2.VideoWriter:
    frame_height, frame_width = frame.shape[:2]

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(state.source_info["fps"]),
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {OUTPUT_VIDEO}"
        )

    state.writer = writer
    state.writer_size = (frame_width, frame_height)

    return writer


# ---------------------------------------------------------------------------
# UNIT EVENT CREATION
# ---------------------------------------------------------------------------

def create_unit_events(
    state: ProcessingState,
    *,
    frame_id: int,
    video_seconds: float,
    timing_source: str,
    count_out: int,
    evidence_frame: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    """
    Create one event for every increment of cumulative count_out.

    A normal increment is one unit. If the counter rises by more than one in a
    single frame, each unit is logged but exact individual cycle times cannot
    be recovered from the aggregate count.
    """
    count_delta = count_out - state.previous_count_out

    if count_delta < 0:
        state.warnings.append(
            {
                "warning_type": "count_out_reset",
                "frame_id": frame_id,
                "video_timestamp_seconds": video_seconds,
                "previous_count_out": state.previous_count_out,
                "current_count_out": count_out,
                "recorded_at_utc": utc_now_iso(),
            }
        )

        state.last_completion_video_seconds = None
        state.latest_cycle_seconds = None
        state.latest_cycle_status = "counter_reset"

        return []

    if count_delta == 0:
        return []

    interval_since_previous_crossing = None

    if state.last_completion_video_seconds is not None:
        interval_since_previous_crossing = max(
            0.0,
            video_seconds - state.last_completion_video_seconds,
        )

    new_events: list[dict[str, Any]] = []

    for position_in_frame in range(1, count_delta + 1):
        state.production_sequence += 1

        event_id = (
            f"{LINE_ID}-unit-{state.production_sequence:06d}"
            f"-frame-{frame_id}"
        )

        if count_delta == 1:
            cycle_seconds = interval_since_previous_crossing
            timing_quality = "exact_frame_crossing"

        else:
            cycle_seconds = None
            timing_quality = "multiple_units_same_frame"

        if cycle_seconds is None:
            if (
                state.last_completion_video_seconds is None
                and count_delta == 1
            ):
                cycle_status = "first_unit"
            else:
                cycle_status = "cycle_unavailable"

            instantaneous_units_per_minute = None

        else:
            cycle_status = (
                "slow"
                if cycle_seconds > SLOW_CYCLE_SECONDS
                else "normal"
            )

            instantaneous_units_per_minute = (
                60.0 / cycle_seconds
                if cycle_seconds > 0
                else None
            )

            state.cycle_values.append(cycle_seconds)

        evidence_path = save_evidence_frame(
            evidence_frame,
            event_id,
        )

        event = {
            "event_id": event_id,
            "event_type": "unit_completed",

            "event_time_utc_processed": utc_now_iso(),

            "video_timestamp_seconds": round(video_seconds, 6),
            "video_timestamp": format_video_time(video_seconds),
            "frame_id": frame_id,

            "timing_source": timing_source,
            "timing_quality": timing_quality,

            "direction": "out",
            "production_sequence": state.production_sequence,

            "position_within_frame_increment": position_in_frame,
            "units_added_in_frame": count_delta,

            "count_out": count_out,

            "cycle_interval_seconds": (
                round(cycle_seconds, 3)
                if cycle_seconds is not None
                else None
            ),

            "batch_interval_seconds": (
                round(interval_since_previous_crossing, 3)
                if (
                    count_delta > 1
                    and interval_since_previous_crossing is not None
                )
                else None
            ),

            "cycle_status": cycle_status,
            "slow_cycle_threshold_seconds": SLOW_CYCLE_SECONDS,

            "instantaneous_units_per_minute": (
                round(instantaneous_units_per_minute, 3)
                if instantaneous_units_per_minute is not None
                else None
            ),

            "facility_id": FACILITY_ID,
            "line_id": LINE_ID,
            "station_id": STATION_ID,
            "camera_id": CAMERA_ID,

            "shift": SHIFT,
            "sku": SKU,
            "work_order": WORK_ORDER,
            "product_name": PRODUCT_NAME,

            "workflow": WORKFLOW,
            "workspace": WORKSPACE,

            "evidence_image_path": evidence_path,
        }

        state.events.append(event)
        new_events.append(event)

        state.latest_cycle_seconds = cycle_seconds
        state.latest_cycle_status = cycle_status

        if state.first_event_video_seconds is None:
            state.first_event_video_seconds = video_seconds

        state.last_event_video_seconds = video_seconds

    state.last_completion_video_seconds = video_seconds

    return new_events


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = "ROBOFLOW_API_KEY"

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Store your replacement "
            "Roboflow API key in this environment variable."
        )

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video does not exist: {INPUT_VIDEO.resolve()}"
        )

    source_info = inspect_video(INPUT_VIDEO)
    state = ProcessingState(source_info=source_info)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    save_live_document(
        state,
        status="initializing",
    )

    client = InferenceHTTPClient.init(
        api_url=API_URL,
        api_key=api_key,
    )

    source = VideoFileSource(
        str(INPUT_VIDEO),
        realtime_processing=False,
    )

    config = StreamConfig(
        stream_output=[],

        # count_in was removed because the Workflow only needs count_out
        # for completed product cycle-time measurement.
        data_output=[
            VIDEO_OUTPUT,
            COUNT_OUT_OUTPUT,
        ],

        realtime_processing=False,
        requested_plan=REQUESTED_PLAN,
        requested_region=REQUESTED_REGION,
        processing_timeout=PROCESSING_TIMEOUT_SECONDS,
    )

    session = client.webrtc.stream(
        source=source,
        workflow=WORKFLOW,
        workspace=WORKSPACE,
        image_input=IMAGE_INPUT,
        config=config,
    )

    @session.on_data
    def on_data(
        data: dict[str, Any],
        metadata: VideoMetadata,
    ) -> None:
        with state.lock:
            frame_id = int(
                getattr(metadata, "frame_id", 0) or 0
            )

            video_seconds, timing_source = metadata_video_seconds(
                metadata,
                fallback_fps=float(source_info["fps"]),
            )

            count_out = extract_counter(
                data.get(COUNT_OUT_OUTPUT),
                COUNT_OUT_OUTPUT,
            )

            # Keep the previous valid counter if this frame omits the field.
            if count_out is None:
                count_out = state.latest_count_out

            workflow_frame = decode_workflow_image(
                data.get(VIDEO_OUTPUT)
            )

            state.processed_frames += 1
            state.latest_frame_id = frame_id
            state.latest_video_seconds = video_seconds
            state.latest_count_out = count_out

            new_events = create_unit_events(
                state,
                frame_id=frame_id,
                video_seconds=video_seconds,
                timing_source=timing_source,
                count_out=count_out,
                evidence_frame=workflow_frame,
            )

            state.previous_count_out = count_out

            summary = build_summary(state)

            if workflow_frame is not None:
                display_frame = add_metrics_panel(
                    workflow_frame,

                    video_seconds=video_seconds,
                    count_out=count_out,

                    latest_cycle_seconds=state.latest_cycle_seconds,
                    average_cycle_seconds=summary["average_cycle_seconds"],
                    median_cycle_seconds=summary["median_cycle_seconds"],

                    estimated_units_per_minute=(
                        summary["estimated_units_per_minute"]
                    ),

                    slow_cycle_count=summary["slow_cycle_count"],
                    cycle_status=state.latest_cycle_status,
                )

                writer = (
                    state.writer
                    or initialize_video_writer(
                        state,
                        display_frame,
                    )
                )

                if state.writer_size != (
                    display_frame.shape[1],
                    display_frame.shape[0],
                ):
                    display_frame = cv2.resize(
                        display_frame,
                        state.writer_size,
                        interpolation=cv2.INTER_AREA,
                    )

                writer.write(display_frame)

                if DISPLAY_PREVIEW:
                    cv2.imshow(
                        PREVIEW_WINDOW_NAME,
                        display_frame,
                    )

                    delay_ms = (
                        max(
                            1,
                            int(round(1000.0 / source_info["fps"])),
                        )
                        if PREVIEW_AT_SOURCE_SPEED
                        else 1
                    )

                    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                        session.close()

            if new_events:
                for event in new_events:
                    print(
                        f'Unit {event["production_sequence"]}: '
                        f'video={event["video_timestamp"]}, '
                        f'count_out={event["count_out"]}, '
                        f'cycle={event["cycle_interval_seconds"]}, '
                        f'status={event["cycle_status"]}'
                    )

                save_live_document(
                    state,
                    status="processing",
                )

            elif (
                JSON_PROGRESS_EVERY_N_FRAMES > 0
                and state.processed_frames
                % JSON_PROGRESS_EVERY_N_FRAMES
                == 0
            ):
                save_live_document(
                    state,
                    status="processing",
                )

    @session.on_error
    def on_error(
        errors: Any,
        metadata: VideoMetadata,
    ) -> None:
        with state.lock:
            warning = {
                "warning_type": "workflow_frame_error",
                "frame_id": int(
                    getattr(metadata, "frame_id", 0) or 0
                ),
                "errors": errors,
                "recorded_at_utc": utc_now_iso(),
            }

            state.warnings.append(warning)

            save_live_document(
                state,
                status="processing_with_errors",
            )

            print(
                f"Workflow error on frame "
                f"{warning['frame_id']}: {errors}"
            )

    try:
        print(f"Input video: {INPUT_VIDEO}")
        print(f"Source FPS: {source_info['fps']:.3f}")
        print("Using count_out as the completed-product counter.")
        print("Press q in the preview window to stop early.")

        session.run()

    except KeyboardInterrupt:
        print("Stopped by user.")
        session.close()

    finally:
        with state.lock:
            if state.writer is not None:
                state.writer.release()
                state.writer = None

            save_live_document(
                state,
                status="completed",
            )

        cv2.destroyAllWindows()

    summary = build_summary(state)

    print("\nDone.")
    print(f"Processed frames: {summary['processed_frames']}")
    print(f"Units logged: {summary['production_units_logged']}")
    print(f"Average cycle: {summary['average_cycle_seconds']} seconds")
    print(
        f"Estimated rate: "
        f"{summary['estimated_units_per_minute']} units/min"
    )
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"JSON log: {JSON_OUTPUT}")
    print(f"Evidence images: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()