import base64
import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata


API_URL = "https://serverless.roboflow.com"
API_KEY = "ROBOFLOW_API_KEY"

WORKSPACE = "tim-4ijf0"
MONITOR_WORKFLOW_ID = "carton-defect-monitor-1790154641066"
OCR_WORKFLOW_ID = "carton-station-ocr-1790324835782"
TRIAGE_WORKFLOW_ID = "carton-defect-triage-1790154653704"

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "package_1.mp4"
OUTPUT_DIRECTORY = BASE_DIR / "carton_monitor_results"
OUTPUT_VIDEO_PATH = OUTPUT_DIRECTORY / f"{VIDEO_PATH.stem}_annotated.mp4"
RESULT_JSON_PATH = OUTPUT_DIRECTORY / f"{VIDEO_PATH.stem}_assessment.json"
EVIDENCE_DIRECTORY = OUTPUT_DIRECTORY / "triage_source_frames"

WINDOW_NAME = "Carton Defect Monitor - Press Q to Exit"

if not API_KEY:
    raise RuntimeError(
        "Set ROBOFLOW_API_KEY to a replacement key first.\n"
        'PowerShell: $env:ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"'
    )
if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

probe = cv2.VideoCapture(str(VIDEO_PATH))
if not probe.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

source_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
source_frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
probe.release()

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY,
).configure(InferenceConfiguration(api_key_transport="header"))

def read_station_from_first_frame():
    record = {
        "workflow": OCR_WORKFLOW_ID,
        "source_frame_index_zero_based": 0,
        "status": "not_started",
        "station_label_text": None,
        "station_id": "UNKNOWN",
        "error": None,
    }

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    first_frame_path = OUTPUT_DIRECTORY / f"{VIDEO_PATH.stem}_station_ocr_first_frame.png"

    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open video for first-frame OCR")

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError("Cannot decode the video's first frame")
        if not cv2.imwrite(str(first_frame_path), first_frame):
            raise RuntimeError(f"Cannot save first frame: {first_frame_path}")

        response = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=OCR_WORKFLOW_ID,
            images={"image": str(first_frame_path)},
            use_cache=False,
        )
        result = response[0] if isinstance(response, list) and response else response
        if not isinstance(result, dict):
            raise ValueError("OCR returned no structured result")

        def value_of(item):
            return item.get("value") if isinstance(item, dict) and "value" in item else item

        raw_text = value_of(result.get("station_label_text"))
        extracted = value_of(result.get("station_id"))
        station_id = str(extracted or "").strip()

        record["station_label_text"] = raw_text
        if station_id:
            record["station_id"] = station_id
            record["status"] = "read"
        else:
            record["status"] = "no_station_id"

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
    finally:
        capture.release()
        first_frame_path.unlink(missing_ok=True)

    print(
        f"[station OCR] status={record['status']} "
        f"text={record['station_label_text']!r} "
        f"station_id={record['station_id']!r}",
        flush=True,
    )
    if record["error"]:
        print(f"[station OCR] {record['error']}", flush=True)
    return record


station_ocr_record = read_station_from_first_frame()
station_id_for_run = station_ocr_record["station_id"]


# Only retrieves original full frames; it does not feed Monitor.
source_capture = cv2.VideoCapture(str(VIDEO_PATH))
if not source_capture.isOpened():
    raise RuntimeError(f"Cannot reopen source video: {VIDEO_PATH}")
source_capture_lock = threading.Lock()

source = VideoFileSource(str(VIDEO_PATH), realtime_processing=False)
config = StreamConfig(

    workflow_parameters={"station_id": station_id_for_run},
    stream_output=[],
    data_output=[
        "output_image",
        "in_zone_package_ids",
        "trigger_package_ids",
        "triage_package_id",
        "package_report",
        "defect_count_current_frame",
        "station_id",
        "mqtt_published",
        "mqtt_error_status",
        "mqtt_message",
    ],
    realtime_processing=False,
    processing_timeout=3600,
)

session = client.webrtc.stream(
    source=source,
    workflow=MONITOR_WORKFLOW_ID,
    workspace=WORKSPACE,
    image_input="image",
    config=config,
)

state_lock = threading.Lock()
display_queue = queue.Queue(maxsize=30)
triage_queue = queue.Queue()

video_writer = None
annotated_frames_received = 0
stopped_early = False
frames_with_results = set()
frames_with_images = set()

# Record entry only after its original full frame is saved.
package_entries_by_id = {}
package_entry_order = []

# Queue Triage at most once per tracker ID.
queued_triage_ids = set()

events = []
triage_results = []
frame_errors = []
session_errors = []
latest_assessment = None


def unwrap(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def ids_from_output(value):
    value = unwrap(value)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    ]


def timestamp_seconds(metadata):
    if metadata.pts is None or metadata.time_base is None:
        return None
    return float(metadata.pts) * float(metadata.time_base)


def original_frame_at(seconds):
    """Retrieve the original, unannotated frame at the event timestamp."""
    if seconds is None:
        raise ValueError(
            "Event has no PTS/time_base; cannot match an original frame"
        )

    index = round(seconds * source_fps)
    if index < 0 or index >= source_frame_count:
        raise ValueError(
            f"Event time {seconds:.4f}s maps outside the source "
            f"({index=}, {source_frame_count=})"
        )

    with source_capture_lock:
        if not source_capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise RuntimeError(f"Cannot seek source video to frame {index}")
        ok, frame = source_capture.read()

    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode original source frame {index}")

    return index, frame


def decode_output_image(value):
    encoded = unwrap(value)
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Missing base64 output_image")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    raw = base64.b64decode(encoded)
    frame = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise ValueError("Cannot decode output_image")
    return frame


def workflow_result(response, workflow_name):
    result = (
        response[0]
        if isinstance(response, list) and response
        else response
    )
    if not isinstance(result, dict):
        raise ValueError(f"{workflow_name} returned no structured result")
    return result


# Display-only state. It does not decide triggers or send integrations.
overlay_status = {
    'triage': 'Waiting for a Monitor trigger',
    'evidence': 'No original frame saved yet',
    'mqtt': 'No publish result received',
    'webhook': 'No webhook result received',
}


def overlay_clock():
    return datetime.now().astimezone().strftime('%H:%M:%S %Z')


def overlay_set(**values):
    with state_lock:
        overlay_status.update(values)


def overlay_bool(value):
    value = unwrap(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() == 'true':
            return True
        if value.strip().lower() == 'false':
            return False
    return None


def overlay_monitor_status(data, frame_id):
    published = overlay_bool(data.get('mqtt_published'))
    error = overlay_bool(data.get('mqtt_error_status'))
    ids = ids_from_output(data.get('trigger_package_ids'))
    # Preserve the last event across ordinary non-event frames.
    if published is True or error is True or ids:
        status = ('PUBLISHED' if published is True else
                  'ERROR' if error is True else 'NOT CONFIRMED')
        package = unwrap(data.get('triage_package_id')) or ', '.join(ids) or '?'
        overlay_set(
            mqtt=f'{status} | package {package} | Monitor #{frame_id}',
            mqtt_time=f'Status received {overlay_clock()}',
            mqtt_detail=str(unwrap(data.get('mqtt_message')) or ''),
        )


def overlay_triage_result(result, job):
    assessment = unwrap(result.get('assessment_json'))
    assessment = assessment if isinstance(assessment, dict) else {}
    route = unwrap(result.get('route')) or assessment.get('route') or 'unknown'
    err = overlay_bool(result.get('ticket_intake_error'))
    if route == 'review_high':
        hook = ('ERROR' if err is True else
                'SUCCESS REPORTED' if err is False else 'STATUS UNAVAILABLE')
    elif route in ('review_low', 'manual_review'):
        hook = 'NOT REQUESTED (route gate)'
    else:
        hook = 'STATUS UNAVAILABLE'
    overlay_set(
        triage=f'RETURNED | package {job["package_id"]} | {route}',
        triage_return=f'Result received {overlay_clock()}',
        assessment_package=job['package_id'],
        severity=str(unwrap(assessment.get('severity')) or 'unknown'),
        defect=str(unwrap(assessment.get('defect_type')) or 'unknown'),
        action=str(unwrap(assessment.get('action')) or 'Not available'),
        webhook=f'{hook} | package {job["package_id"]}',
        webhook_time=f'Status received {overlay_clock()}',
        webhook_detail=str(unwrap(result.get('ticket_intake_message')) or ''),
    )


def assessment_overlay(frame, frame_id=None, seconds=None, data=None):
    """Append a colored status panel to visualization only, never evidence."""
    data = data if isinstance(data, dict) else {}
    with state_lock:
        ui = dict(overlay_status)
        captured = len(package_entries_by_id)
        queued = len(queued_triage_ids)
        completed = len(triage_results)
        error_count = len(frame_errors) + len(session_errors)
    height, width = frame.shape[:2]
    panel_width = 620
    canvas_height = max(height, 780)
    output = np.full((canvas_height, width + panel_width, 3), (22, 18, 30), dtype=np.uint8)
    output[:height, :width] = frame
    x, y = width + 20, 28
    max_text_width = panel_width - 40
    white, muted = (245, 245, 250), (180, 175, 195)
    purple, cyan = (245, 155, 200), (235, 220, 90)
    green, amber, red = (145, 230, 145), (80, 205, 255), (110, 110, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def row(value, color=white, scale=0.56, max_lines=1):
        nonlocal y
        value = str(value).replace('\n', ' ').replace('\r', ' ')
        value = value.encode('ascii', 'replace').decode('ascii')
        words = value.split()
        lines, current = [], ''
        for word in words:
            proposed = f'{current} {word}'.strip()
            if cv2.getTextSize(proposed, font, scale, 1)[0][0] > max_text_width and current:
                lines.append(current)
                current = word
            else:
                current = proposed
        if current:
            lines.append(current)
        for i, line in enumerate(lines[:max_lines]):
            if i == max_lines - 1 and len(lines) > max_lines:
                line += ' ...'
            while cv2.getTextSize(line, font, scale, 1)[0][0] > max_text_width:
                line = line[:-5] + ' ...' if len(line) > 5 else ''
            cv2.putText(output, line, (x, y), font, scale, color, 1, cv2.LINE_AA)
            y += 23

    def section(label, color=purple):
        nonlocal y
        y += 10
        cv2.line(output, (x, y - 3), (width + panel_width - 20, y - 3), (65, 53, 80), 1)
        y += 20
        row(label, color, 0.62)

    row('CARTON INSPECTION / LIVE STATUS', purple, 0.69)
    row(f'Station: {station_id_for_run} | OCR: {station_ocr_record["status"]}', cyan)
    index = round(seconds * source_fps) if seconds is not None else None
    position = f'{index + 1}/{source_frame_count}' if index is not None else 'unavailable'
    video_time = f'{seconds:.3f}s' if seconds is not None else 'unavailable'
    row(f'Source frame: {position} | Video time: {video_time}')
    row(f'Monitor frame ID: {frame_id} | Local: {overlay_clock()}', muted)
    ids = ids_from_output(data.get('in_zone_package_ids'))
    triggers = ids_from_output(data.get('trigger_package_ids'))
    row(f'Qualified IDs: {", ".join(ids) or "none"}', cyan)
    row(f'Trigger IDs: {", ".join(triggers) or "none"} | Damage boxes: {unwrap(data.get("defect_count_current_frame"))}', amber)
    row(f'Saved: {captured} | Queued total: {queued} | Returned: {completed} | Errors: {error_count}', muted)
    section('EVIDENCE / TRIAGE', cyan)
    row(ui['evidence'], muted)
    row(ui['triage'], amber if ui['triage'].startswith('REQUEST') else white, max_lines=2)
    row(ui.get('triage_send', 'No Triage request started'), cyan, max_lines=2)
    row(ui.get('triage_return', 'Awaiting assessment result'), muted)
    severity = ui.get('severity', 'pending')
    row(f'Assessment P{ui.get("assessment_package", "?")}: {severity} | Defect: {ui.get("defect", "pending")}', red if severity == 'high' else green if severity == 'low' else amber, max_lines=2)
    row(f'Operator action: {ui.get("action", "pending")}', white, max_lines=2)
    section('MQTT / MONITOR EVENT', purple)
    row(ui['mqtt'], green if ui['mqtt'].startswith('PUBLISHED') else red if ui['mqtt'].startswith('ERROR') else amber, max_lines=2)
    row(ui.get('mqtt_time', 'Waiting for Workflow status'), muted)
    row(ui.get('mqtt_detail', ''), muted)
    section('WEBHOOK / TRIAGE RESULT', purple)
    row(ui['webhook'], green if ui['webhook'].startswith('SUCCESS') else red if ui['webhook'].startswith('ERROR') else amber, max_lines=2)
    row(ui.get('webhook_time', 'Status arrives with the Triage response'), muted)
    row(ui.get('webhook_detail', ''), muted)
    return output


def fit_for_display(frame, max_width=1280, max_height=800):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def read_station_for_package(job):
    """
    Call the independent OCR Workflow on the original entry frame.

    An OCR error or unreadable label returns UNKNOWN. This function never
    decides whether the package is damaged or whether Gemini should run.
    """
    package_id = job["package_id"]
    record = {
        "workflow": OCR_WORKFLOW_ID,
        "status": "not_started",
        "station_label_text": None,
        "station_id": "UNKNOWN",
        "error": None,
    }

    try:
        response = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=OCR_WORKFLOW_ID,
            images={"image": job["source_frame_path"]},
            use_cache=False,
        )
        result = workflow_result(response, "OCR")
        raw_text = unwrap(result.get("station_label_text"))
        extracted = unwrap(result.get("station_id"))
        station_id = str(extracted or "").strip()

        record["station_label_text"] = raw_text
        if station_id:
            record["status"] = "read"
            record["station_id"] = station_id
            print(
                f"[ocr {package_id}] read {raw_text!r}; "
                f"station_id={station_id!r}",
                flush=True,
            )
        else:
            record["status"] = "no_station_id"
            print(
                f"[ocr {package_id}] text={raw_text!r}; "
                "no station ID found, using UNKNOWN",
                flush=True,
            )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        print(
            f"[ocr {package_id} failed] {exc!r}; "
            "continuing to Triage with UNKNOWN",
            flush=True,
        )

    return record


def triage_worker():
    """
    For each Monitor-authorized damaged package:
      1. Try OCR independently.
      2. Call the saved Triage Workflow regardless of OCR outcome.
    Triage owns Gemini and the webhook; Python calls neither directly.
    """
    global latest_assessment

    while True:
        job = triage_queue.get()
        try:
            if job is None:
                return

            result_record = {
                **job,
                "status": "started",
                "ocr": None,
                "station_id_used_for_triage": "UNKNOWN",
                "route": None,
                "assessment": None,
                "webhook_error": None,
                "webhook_message": None,
                "gemini_error": None,
                "json_parse_error": None,
                "error": None,
            }

            # OCR has its own exception handling. No text or no station
            # code is not permission to skip Gemini Triage.
            result_record["ocr"] = dict(station_ocr_record)
            station_id = station_id_for_run
            result_record["station_id_used_for_triage"] = station_id

            try:
                overlay_set(
                    triage=f'REQUEST STARTED | package {job["package_id"]}',
                    triage_send=(f'Evidence frame #{job["source_frame_index_zero_based"] + 1} '
                                 f'request started {overlay_clock()} | trigger Monitor #{job["monitor_frame_id"]}'),
                    triage_return='Waiting for the Workflow response',
                )
                response = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=TRIAGE_WORKFLOW_ID,
                    images={"image": job["source_frame_path"]},
                    parameters={
                        "station_id": station_id,
                        "package_id": job["package_id"],
                    },
                    use_cache=False,
                )
                result = workflow_result(response, "Triage")
                overlay_triage_result(result, job)

                result_record.update({
                    "status": "completed",
                    "route": result.get("route"),
                    "assessment": result.get("assessment_json"),
                    "webhook_error": result.get("ticket_intake_error"),
                    "webhook_message": result.get("ticket_intake_message"),
                    "gemini_error": result.get("gemini_error"),
                    "json_parse_error": result.get("json_parse_error"),
                })

                assessment = result.get("assessment_json")
                if isinstance(assessment, dict):
                    with state_lock:
                        latest_assessment = {
                            **assessment,
                            "package_id": job["package_id"],
                            "route": result.get("route"),
                        }

                print(
                    f"[triage {job['package_id']}] "
                    f"station_id={station_id!r} "
                    f"route={result_record['route']} "
                    f"webhook_error={result_record['webhook_error']} "
                    f"webhook_message={result_record['webhook_message']}",
                    flush=True,
                )
            except Exception as exc:
                overlay_set(triage=f'FAILED | package {job["package_id"]}', triage_return=f'Failure received {overlay_clock()}')
                result_record["status"] = "failed"
                result_record["error"] = repr(exc)
                print(
                    f"[triage {job['package_id']} failed] {exc!r}",
                    flush=True,
                )

            with state_lock:
                triage_results.append(result_record)
        finally:
            triage_queue.task_done()


def process_workflow_data(data: dict, metadata: VideoMetadata):
    global video_writer, annotated_frames_received

    frame_id = int(metadata.frame_id)

    # A failed Workflow frame can invoke this callback with data=None.
    # Do this BEFORE checking membership or calling data.get(...).
    if not isinstance(data, dict):
        print(
            f"[frame {frame_id}] no structured Monitor data",
            flush=True,
        )
        return

    seconds = timestamp_seconds(metadata)
    rounded_seconds = round(seconds, 4) if seconds is not None else None

    with state_lock:
        frames_with_results.add(frame_id)

    in_zone_ids = ids_from_output(data.get("in_zone_package_ids"))
    trigger_ids = ids_from_output(data.get("trigger_package_ids"))
    report = unwrap(data.get("package_report"))
    if not isinstance(report, dict):
        report = {}

    # 1. Capture the first successfully received, qualified in-zone
    # original frame for EVERY package, including undamaged packages.
    for package_id in dict.fromkeys(in_zone_ids):
        if package_id not in package_entries_by_id:
            try:
                source_index, original = original_frame_at(seconds)
                evidence_path = EVIDENCE_DIRECTORY / (
                    f"{VIDEO_PATH.stem}_entry_frame_{source_index:06d}_"
                    f"package_{package_id}.png"
                )
                if not cv2.imwrite(str(evidence_path), original):
                    raise RuntimeError(
                        f"Cannot save entry frame: {evidence_path}"
                    )

                entry = {
                    "package_id": package_id,
                    "entry_monitor_frame_id": frame_id,
                    "entry_video_time_seconds": rounded_seconds,
                    "source_frame_index_zero_based": source_index,
                    "source_frame_path": str(evidence_path),
                    "defect_count_estimate": 0,
                    "triage_queued": False,
                }
                package_entries_by_id[package_id] = entry
                package_entry_order.append(package_id)
                overlay_set(evidence=f'Saved package {package_id}: source frame #{source_index + 1}')

                print(
                    f"[frame {frame_id}] captured in-zone package "
                    f"{package_id}: {evidence_path}",
                    flush=True,
                )
            except Exception as exc:
                message = (
                    f"Frame {frame_id}: package {package_id} "
                    f"entry capture failed: {exc!r}"
                )
                with state_lock:
                    frame_errors.append(message)
                print("[entry error]", message, flush=True)
                # Leave uncaptured so a later in-zone frame can retry.
                continue

        # Reporting only: do not infer Triage eligibility from this count.
        row = report.get(package_id)
        if isinstance(row, dict):
            entry = package_entries_by_id[package_id]
            entry["defect_count_estimate"] = row.get(
                "defect_count_estimate",
                entry["defect_count_estimate"],
            )
            entry["last_reported_zone_seconds"] = row.get("zone_seconds")
            entry["last_reported_defects_visible"] = row.get(
                "defects_visible"
            )

    # 2. Only Monitor's damaged-only trigger can queue Triage.
    if not trigger_ids and not unwrap(data.get("mqtt_published")):
        return

    event = {
        "frame_id": frame_id,
        "video_time_seconds": rounded_seconds,
        "in_zone_package_ids": in_zone_ids,
        "trigger_package_ids": trigger_ids,
        "triage_package_id_from_monitor": unwrap(
            data.get("triage_package_id")
        ),
        "defect_count_current_frame": unwrap(
            data.get("defect_count_current_frame")
        ),
        "package_report": report,
        "monitor_station_id": unwrap(data.get("station_id")),
        "mqtt_published": unwrap(data.get("mqtt_published")),
        "mqtt_error_status": unwrap(data.get("mqtt_error_status")),
        "mqtt_message": unwrap(data.get("mqtt_message")),
        "new_triage_ids": [],
        "duplicate_triage_ids": [],
        "handoff_errors": [],
    }

    for package_id in dict.fromkeys(trigger_ids):
        if package_id in queued_triage_ids:
            event["duplicate_triage_ids"].append(package_id)
            continue

        entry = package_entries_by_id.get(package_id)
        if entry is None:
            message = (
                f"Package {package_id}: Monitor triggered but no original "
                "in-zone entry frame was successfully captured"
            )
            event["handoff_errors"].append(message)
            print("[handoff error]", message, flush=True)
            continue

        try:
            triage_queue.put({
                "package_id": package_id,
                "monitor_frame_id": frame_id,
                "entry_monitor_frame_id": entry[
                    "entry_monitor_frame_id"
                ],
                "source_frame_index_zero_based": entry[
                    "source_frame_index_zero_based"
                ],
                "video_time_seconds": rounded_seconds,
                "defect_count_estimate_at_trigger": (
                    report.get(package_id, {}).get(
                        "defect_count_estimate"
                    )
                    if isinstance(report.get(package_id), dict)
                    else None
                ),
                "source_frame_path": entry["source_frame_path"],
            })
            queued_triage_ids.add(package_id)
            entry["triage_queued"] = True
            event["new_triage_ids"].append(package_id)

            print(
                f"[frame {frame_id}] queued damaged package "
                f"{package_id} for independent OCR and Triage",
                flush=True,
            )
        except Exception as exc:
            message = f"Package {package_id}: {exc!r}"
            event["handoff_errors"].append(message)
            print("[handoff error]", message, flush=True)

    with state_lock:
        events.append(event)


def render_monitor_video(data, metadata):
    global video_writer, annotated_frames_received
    if not isinstance(data, dict):
        return
    frame_id = int(metadata.frame_id)
    seconds = timestamp_seconds(metadata)
    # A missing or undecodable visualization does not hide a valid
    # structured zone output. Handle entry and trigger separately.
    if data.get("output_image") is not None:
        try:
            frame = decode_output_image(data["output_image"])
            shown = assessment_overlay(frame, frame_id, seconds, data)

            with state_lock:
                if video_writer is None:
                    height, width = shown.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(OUTPUT_VIDEO_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Cannot create video: {OUTPUT_VIDEO_PATH}"
                        )
                video_writer.write(shown)
                annotated_frames_received += 1
                frames_with_images.add(frame_id)

            try:
                display_queue.put_nowait(shown)
            except queue.Full:
                try:
                    display_queue.get_nowait()
                    display_queue.put_nowait(shown)
                except queue.Empty:
                    pass
        except Exception as exc:
            message = f"Frame {frame_id}: annotated image: {exc!r}"
            with state_lock:
                frame_errors.append(message)
            print("[frame error]", message, flush=True)


@session.on_data()
def handle_workflow_data(data: dict, metadata: VideoMetadata):
    if isinstance(data, dict):
        overlay_monitor_status(data, int(metadata.frame_id))
    try:
        process_workflow_data(data, metadata)
    finally:
        render_monitor_video(data, metadata)


@session.on_error
def handle_frame_error(errors, metadata: VideoMetadata):
    message = f"Frame {metadata.frame_id}: {errors}"
    with state_lock:
        frame_errors.append(message)
    print("[workflow error]", message, flush=True)


def run_session():
    try:
        session.run()
    except Exception as exc:
        with state_lock:
            session_errors.append(repr(exc))


print(f"Video:             {VIDEO_PATH}")
print(f"Monitor Workflow:  {MONITOR_WORKFLOW_ID}")
print(f"Station OCR:       {OCR_WORKFLOW_ID}")
print(f"Triage Workflow:   {TRIAGE_WORKFLOW_ID}")
print(f"Source frames:     {source_frame_count}")
print(f"Annotated output:  {OUTPUT_VIDEO_PATH}")
print(f"JSON report:       {RESULT_JSON_PATH}")
print("Press Q to stop early.")

triage_thread = threading.Thread(target=triage_worker, daemon=True)
triage_thread.start()
worker = threading.Thread(target=run_session, daemon=True)
worker.start()

show_window = True
try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
except cv2.error:
    show_window = False
    print("[warning] No graphical display; continuing without a window.")

try:
    while worker.is_alive() or not display_queue.empty():
        try:
            frame = display_queue.get(timeout=0.05)
            if show_window:
                cv2.imshow(WINDOW_NAME, fit_for_display(frame))
        except queue.Empty:
            pass

        if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
            stopped_early = True
            session.close()
            break
except KeyboardInterrupt:
    stopped_early = True
    session.close()
finally:
    worker.join(timeout=15)
    if worker.is_alive():
        with state_lock:
            session_errors.append(
                "WebRTC session did not finish within 15 seconds."
            )

    try:
        session.close()
    except Exception:
        pass

    # Finish queued OCR + Triage work before writing the JSON report.
    triage_queue.put(None)
    triage_queue.join()
    triage_thread.join()

    with source_capture_lock:
        source_capture.release()

    with state_lock:
        if video_writer is not None:
            video_writer.release()

        final_report = {
            "video": str(VIDEO_PATH),
            "monitor_workflow": MONITOR_WORKFLOW_ID,
            "ocr_workflow": OCR_WORKFLOW_ID,
            "triage_workflow": TRIAGE_WORKFLOW_ID,
            "source_frame_count_reported": source_frame_count,
            "annotated_frames_received": annotated_frames_received,
            "frames_with_structured_results": len(frames_with_results),
            "distinct_frame_ids_with_images": len(frames_with_images),
            "stopped_early": stopped_early,
            "package_entries": [
                package_entries_by_id[package_id]
                for package_id in package_entry_order
            ],
            "events": list(events),
            "triage_results": list(triage_results),
            "frame_errors": list(frame_errors),
            "session_errors": list(session_errors),
            "station_ocr_preflight": dict(station_ocr_record),
            "station_id_supplied_to_monitor": station_id_for_run,
            "note": (
                "package_entries records the first successfully received "
                "qualified in-zone frame per tracked ID, not necessarily "
                "the true first qualifying frame if Monitor failed earlier. "
                "defect_count_estimate counts associated image regions, "
                "not verified physical defects. OCR failure does not skip "
                "Triage. A successful Gemini result is not proof of "
                "webhook delivery."
            ),
        }

    RESULT_JSON_PATH.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if show_window:
        cv2.destroyAllWindows()

print(f"Annotated video: {OUTPUT_VIDEO_PATH}")
print(f"JSON report:     {RESULT_JSON_PATH}")
print(
    f"Received {annotated_frames_received} annotated frames "
    f"from {source_frame_count} reported source frames."
)
print(f"Packages captured: {len(package_entries_by_id)}")
print(f"Damaged packages queued for Triage: {len(queued_triage_ids)}")
if frame_errors or session_errors or stopped_early:
    print("Run had errors or stopped early; inspect the JSON report.")
