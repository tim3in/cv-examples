import math

def run(self, predictions, image_id, width, height,
        defect_count, selected, reason):
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("Invalid original image dimensions")

    n = len(predictions)
    if n != int(defect_count):
        raise ValueError("Detection count mismatch")

    names = list(predictions.data.get("class_name", []))
    scores = predictions.confidence
    allowed = {
        "fabric_broken_pick", "fabric_hole",
        "fabric_missing_pick", "fabric_thick_weft"
    }
    rows = []
    if n > 0 and (len(names) != n or scores is None or len(scores) != n):
        raise ValueError("Missing prediction classes or confidence scores")

    for i, box in enumerate(predictions.xyxy):
        name = str(names[i])
        confidence = float(scores[i])
        coords = [float(v) for v in box]
        if (name not in allowed or not math.isfinite(confidence)
                or not all(math.isfinite(v) for v in coords)):
            raise ValueError("Invalid defect prediction")
        rows.append({
            "class": name,
            "confidence": confidence,
            "bbox": {
                "x_min": coords[0], "y_min": coords[1],
                "x_max": coords[2], "y_max": coords[3]
            }
        })

    return {
        "result": {
            "processing_status": "success",
            "image_id": str(image_id),
            "selected_for_review": bool(selected),
            "reason": str(reason),
            "defect_count": n,
            "image_dimensions": {"width": w, "height": h},
            "coordinate_format": {
                "format": "xyxy", "units": "pixels",
                "reference": "original_image", "origin": "top_left",
                "x_direction": "right", "y_direction": "down"
            },
            "predictions": rows
        }
    }
