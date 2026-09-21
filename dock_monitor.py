import os
import cv2
import json
import base64
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = "ROBOFLOW_API_KEY"
API_URL = "https://serverless.roboflow.com"

WORKSPACE = "tim-4ijf0"

# Main RF-DETR + ByteTrack + Time-in-Zone Workflow
VIDEO_WORKFLOW_ID = "dock-door-monitor"

# Separate Gemini image-analysis Workflow
GEMINI_WORKFLOW_ID = "dock-door-frame-analyzer-1789574668417"

# Video must be in the same directory as this script
VIDEO_PATH = "door_4.mp4"

# Preserve the Workflow's configured default
DWELL_SECONDS = 1

OUTPUT_DIRECTORY = Path("dock_monitor_results")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIRECTORY / "door_annotated.mp4"
RESULT_JSON_PATH = OUTPUT_DIRECTORY / "gemini_assessment.json"

WINDOW_NAME = "Dock Door Monitor - Press Q to Exit"


if API_KEY == "YOUR_ROBOFLOW_API_KEY":
    raise RuntimeError(
        "Set your API key first.\n\n"
        "macOS/Linux:\n"
        '  export ROBOFLOW_API_KEY="YOUR_API_KEY"\n\n'
        "Windows PowerShell:\n"
        '  $env:ROBOFLOW_API_KEY="YOUR_API_KEY"\n'
    )

if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(
        f"Could not find {VIDEO_PATH}. Put door.mp4 in the same folder "
        "as dock_monitor.py."
    )


# ============================================================
# ROBOFLOW CLIENTS
# ============================================================

video_client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY,
).configure(
    InferenceConfiguration(api_key_transport="header")
)

gemini_client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY,
).configure(
    InferenceConfiguration(api_key_transport="header")
)


# ============================================================
# APPLICATION STATE
# ============================================================

state_lock = threading.Lock()

latest_assessment = None
trigger_received = False
gemini_future = None

gemini_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="gemini-analysis",
)

video_writer = None


# ============================================================
# IMAGE DECODING
# ============================================================

def find_base64_image(value):
    """
    Recursively locate a serialized image in a Workflow output.

    Supports:
      - Base64 strings
      - data:image/... strings
      - {"value": "..."} wrappers
      - {"image": "..."} wrappers
    """
    if value is None:
        return None

    if isinstance(value, str):
        if not value or value == "[image omitted from JSONL]":
            return None
        return value

    if isinstance(value, dict):
        preferred_keys = (
            "value",
            "image",
            "base64",
            "data",
        )

        for key in preferred_keys:
            if key in value:
                result = find_base64_image(value[key])
                if result:
                    return result

        for nested_value in value.values():
            result = find_base64_image(nested_value)
            if result:
                return result

        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            result = find_base64_image(item)
            if result:
                return result

    return None


def decode_workflow_image(value):
    """Convert a serialized Workflow image to a BGR NumPy array."""
    encoded = find_base64_image(value)

    if encoded is None:
        return None

    try:
        if encoded.startswith("data:image"):
            encoded = encoded.split(",", 1)[1]

        encoded += "=" * (-len(encoded) % 4)

        raw_bytes = base64.b64decode(encoded)
        array = np.frombuffer(raw_bytes, dtype=np.uint8)

        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    except Exception as error:
        print(f"[warning] Could not decode Workflow image: {error}")
        return None


# ============================================================
# GEMINI RESULT HELPERS
# ============================================================

def normalize_gemini_result(result):
    """
    Normalize the result returned by run_workflow().
    """
    if isinstance(result, list):
        if not result:
            return {}
        result = result[0]

    if not isinstance(result, dict):
        return {
            "gemini_raw_output": str(result),
        }

    return result


def run_gemini_analysis(trigger_image, source_frame_id, video_time_seconds):
    """
    Send the one-time trigger frame to the separate Gemini Workflow.
    """
    global latest_assessment

    try:
        print()
        print("============================================================")
        print("TRUCK DWELL TRIGGER RECEIVED")
        print("============================================================")
        print(f"Frame:       {source_frame_id}")
        print(f"Video time:  {video_time_seconds:.3f} seconds")
        print("Running Dock Door Frame Analyzer...")
        print("============================================================")

        result = gemini_client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=GEMINI_WORKFLOW_ID,
            images={
                "image": trigger_image,
            },
            use_cache=False,
        )

        output = normalize_gemini_result(result)

        assessment = {
            "source_frame_id": source_frame_id,
            "video_time_seconds": round(video_time_seconds, 4),
            "dock_status": output.get("dock_status"),
            "blocked_by": output.get("blocked_by"),
            "assessment_reason": output.get("assessment_reason"),
            "json_parse_error": output.get("json_parse_error"),
            "gemini_raw_output": output.get("gemini_raw_output"),
        }

        with state_lock:
            latest_assessment = assessment

        with RESULT_JSON_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                assessment,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print()
        print("============================================================")
        print("GEMINI DOCK ASSESSMENT")
        print("============================================================")
        print(f"Dock status: {assessment['dock_status']}")
        print(f"Blocked by:  {assessment['blocked_by']}")
        print(f"Reason:      {assessment['assessment_reason']}")
        print(f"Parse error: {assessment['json_parse_error']}")
        print(f"Result file: {RESULT_JSON_PATH}")
        print("============================================================")
        print()

        return assessment

    except Exception as error:
        assessment = {
            "source_frame_id": source_frame_id,
            "video_time_seconds": round(video_time_seconds, 4),
            "dock_status": "analysis_error",
            "blocked_by": None,
            "assessment_reason": str(error),
            "json_parse_error": True,
            "gemini_raw_output": None,
        }

        with state_lock:
            latest_assessment = assessment

        with RESULT_JSON_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                assessment,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("============================================================")
        print("GEMINI ANALYSIS FAILED")
        print("============================================================")
        print(str(error))
        print("============================================================")
        print()

        return assessment


# ============================================================
# VIDEO OVERLAY
# ============================================================

def safe_text(value, fallback="none"):
    if value is None:
        return fallback

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    text = str(value).strip()
    return text if text else fallback


def wrap_text(text, max_characters=75):
    words = text.split()

    if not words:
        return [""]

    lines = []
    current_line = ""

    for word in words:
        candidate = (
            f"{current_line} {word}".strip()
            if current_line
            else word
        )

        if len(candidate) <= max_characters:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def draw_assessment_overlay(frame):
    """
    Draw the current Gemini state over the annotated Workflow frame.
    """
    with state_lock:
        assessment = (
            dict(latest_assessment)
            if latest_assessment is not None
            else None
        )
        has_trigger = trigger_received

    output = frame.copy()

    if assessment is not None:
        status = safe_text(
            assessment.get("dock_status"),
            fallback="unknown",
        )
        blocked_by = safe_text(
            assessment.get("blocked_by"),
            fallback="none",
        )
        reason = safe_text(
            assessment.get("assessment_reason"),
            fallback="No reason returned",
        )

        lines = [
            f"Status: {status}",
            f"Blocked by: {blocked_by}",
        ]

        reason_lines = wrap_text(
            f"Reason: {reason}",
            max_characters=78,
        )
        lines.extend(reason_lines)

    elif has_trigger:
        lines = [
            "Status: analyzing trigger frame...",
        ]

    else:
        lines = [
            f"Truck dwell threshold: {DWELL_SECONDS} second(s)",
            "Status: waiting for truck trigger",
        ]

    # Supervision-style labels: clean, thin text on padded backgrounds.
    # Uses existing OpenCV dependencies; no extra installation is required.
    height, width = output.shape[:2]
    ui_scale = max(0.45, min(width / 1280.0, height / 720.0, 2.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58 * ui_scale
    font_thickness = max(1, round(ui_scale))
    padding = max(5, round(12 * ui_scale))
    margin = max(4, round(16 * ui_scale))
    gap = max(3, round(6 * ui_scale))
    (_, text_height), baseline = cv2.getTextSize(
        "Ag", font, font_scale, font_thickness
    )
    line_height = text_height + baseline + gap
    header_height = line_height + padding * 2
    panel_width = min(round(660 * ui_scale), width - margin * 2)
    text_width = panel_width - padding * 2

    # Very small frames cannot hold a readable information panel.
    if text_width < 30 or height < margin * 2 + header_height + padding * 2 + line_height:
        return output

    # OpenCV expects BGR colors.
    purple = (206, 6, 103)  # #6706CE
    background = (30, 26, 24)
    foreground = (245, 245, 245)
    muted = (198, 190, 184)
    status_color = (112, 92, 72)  # Waiting / unknown
    if assessment is not None:
        normalized_status = status.lower().replace("-", "_").replace(" ", "_")
        if "error" in normalized_status or assessment.get("json_parse_error"):
            status_color = (65, 55, 190)
        elif normalized_status in {"clear", "available", "unblocked", "not_blocked", "dock_clear", "dock_available"}:
            status_color = (95, 120, 20)
        elif "blocked" in normalized_status or "occupied" in normalized_status:
            status_color = (65, 55, 190)
    elif has_trigger:
        status_color = purple

    def measured_width(text):
        return cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]

    def fit_lines(text):
        # Pixel-based wrapping also handles long words and JSON values.
        wrapped = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}" if current else word
            if measured_width(candidate) <= text_width:
                current = candidate
                continue
            if current:
                wrapped.append(current)
            current = ""
            for character in word:
                if current and measured_width(current + character) > text_width:
                    wrapped.append(current)
                    current = ""
                current += character
        wrapped.append(current)
        return wrapped

    def ellipsize(text):
        while text and measured_width(text + "...") > text_width:
            text = text[:-1]
        return text.rstrip() + "..."

    # Keep existing assessment content and state handling unchanged.
    # Rejoin the original character-wrapped reason before pixel wrapping.
    display_lines = lines[:2] + [" ".join(lines[2:])] if len(lines) > 2 else lines
    rows = []
    for line in display_lines:
        is_status = line.startswith("Status:")
        rows.extend((part, is_status) for part in fit_lines(line))

    max_rows = max(1, (
        height - margin * 2 - header_height - padding * 2
    ) // line_height)
    if len(rows) > max_rows:
        rows = rows[:max_rows]
        rows[-1] = (ellipsize(rows[-1][0]), rows[-1][1])

    panel_height = header_height + padding * 2 + len(rows) * line_height
    x1, y1 = margin, margin
    x2, y2 = x1 + panel_width, y1 + panel_height

    # Blend only the panel region, leaving the rest of the video untouched.
    panel = output[y1:y2, x1:x2]
    tint = np.full_like(panel, background)
    cv2.addWeighted(tint, 0.90, panel, 0.10, 0, panel)
    cv2.rectangle(output, (x1, y1), (x2 - 1, y1 + header_height - 1), purple, -1)
    title = "DOCK ASSESSMENT"
    if measured_width(title) > text_width:
        title = ellipsize(title)
    cv2.putText(
        output, title, (x1 + padding, y1 + padding + text_height),
        font, font_scale, foreground, font_thickness, cv2.LINE_AA,
    )

    for index, (line, is_status) in enumerate(rows):
        row_top = y1 + header_height + padding + index * line_height
        if is_status:
            # A solid status label, as used in Supervision annotations.
            cv2.rectangle(
                output, (x1 + padding - 4, row_top - 2),
                (min(x2 - 1, x1 + padding + measured_width(line) + 4),
                 row_top + text_height + baseline + 2),
                status_color, -1,
            )
        cv2.putText(
            output, line, (x1 + padding, row_top + text_height),
            font, font_scale, foreground if is_status else muted,
            font_thickness, cv2.LINE_AA,
        )

    return output


def fit_for_display(frame, max_width=1280, max_height=800):
    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale >= 1:
        return frame

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# VIDEO SOURCE AND WEBRTC CONFIGURATION
# ============================================================

source = VideoFileSource(
    VIDEO_PATH,
    realtime_processing=False,
)

config = StreamConfig(
    # Continuous annotated video from the main Workflow.
    stream_output=[
        "output_image",
    ],

    # Sparse trigger outputs delivered through the WebRTC data channel.
    data_output=[
        "trigger_frame",
        "trucks_over_dwell_threshold",
    ],

    # Keep the configured default dwell threshold.
    workflow_parameters={
        "dwell_seconds": DWELL_SECONDS,
    },

    # Process every frame of door.mp4.
    realtime_processing=False,

    processing_timeout=3600,

    # Optional hosted configuration:
    # requested_plan="webrtc-gpu-medium",
    # requested_region="us",
)

session = video_client.webrtc.stream(
    source=source,
    workflow=VIDEO_WORKFLOW_ID,
    workspace=WORKSPACE,
    image_input="image",
    config=config,
)


# ============================================================
# WEBRTC CALLBACKS
# ============================================================

@session.on_frame
def handle_video_frame(frame, metadata):
    """
    Receive the continuous annotated output_image through WebRTC.
    """
    global video_writer

    output_frame = draw_assessment_overlay(frame)

    if video_writer is None:
        height, width = output_frame.shape[:2]

        capture = cv2.VideoCapture(VIDEO_PATH)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        video_writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

    video_writer.write(output_frame)

    cv2.imshow(
        WINDOW_NAME,
        fit_for_display(output_frame),
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()


@session.on_data()
def handle_workflow_data(data: dict, metadata: VideoMetadata):
    """
    Receive the sparse trigger_frame output.
    """
    global trigger_received
    global gemini_future

    if not data:
        return

    trigger_image = decode_workflow_image(
        data.get("trigger_frame")
    )

    if trigger_image is None:
        return

    # Additional application-level one-time guard.
    with state_lock:
        if trigger_received:
            return

        trigger_received = True

    frame_id = metadata.frame_id

    if (
        metadata.pts is not None
        and metadata.time_base is not None
    ):
        video_time_seconds = (
            float(metadata.pts) *
            float(metadata.time_base)
        )
    else:
        video_time_seconds = 0.0

    trigger_path = (
        OUTPUT_DIRECTORY /
        f"trigger_frame_{frame_id}.jpg"
    )

    cv2.imwrite(
        str(trigger_path),
        trigger_image,
    )

    print(f"Saved trigger frame: {trigger_path}")

    # Run Gemini in a separate thread so video callbacks are not blocked.
    gemini_future = gemini_executor.submit(
        run_gemini_analysis,
        trigger_image.copy(),
        frame_id,
        video_time_seconds,
    )


@session.on_error
def handle_frame_error(errors, metadata: VideoMetadata):
    print(
        f"[workflow error] Frame {metadata.frame_id}: {errors}"
    )


# ============================================================
# RUN
# ============================================================

try:
    print()
    print("============================================================")
    print("DOCK DOOR MONITOR")
    print("============================================================")
    print(f"Video:             {VIDEO_PATH}")
    print(f"Video Workflow:    {VIDEO_WORKFLOW_ID}")
    print(f"Gemini Workflow:   {GEMINI_WORKFLOW_ID}")
    print(f"Dwell threshold:   {DWELL_SECONDS} second(s)")
    print(f"Output directory:  {OUTPUT_DIRECTORY}")
    print("Press Q in the video window to stop.")
    print("============================================================")
    print()

    session.run()

    # The video can finish before the Gemini request completes.
    if gemini_future is not None:
        print("Waiting for Gemini assessment to finish...")
        gemini_future.result()

    with state_lock:
        final_assessment = (
            dict(latest_assessment)
            if latest_assessment is not None
            else None
        )

    if final_assessment is None:
        print()
        print("Video processing finished.")
        print("No truck dwell trigger was emitted.")
    else:
        print()
        print("Video processing and Gemini analysis finished.")

    print(f"Annotated video: {OUTPUT_VIDEO_PATH}")

finally:
    try:
        session.close()
    except Exception:
        pass

    gemini_executor.shutdown(wait=True)

    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()
