import json
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["ROBOFLOW_API_KEY"],
).configure(InferenceConfiguration(
    api_key_transport="header",
    confidence_threshold=0.20,  # Replace with your validated cutoff.
))

result = client.infer(
    "inspection.jpg",
    model_id="YOUR_RFDETR_SMALL_MODEL_ID",
)
predictions = result["predictions"]
screws = [p for p in predictions if p["class"] == "screw"]
expected_count = 4
screw_count = len(screws)

print(json.dumps({
    "expected_count": expected_count,
    "screw_count": screw_count,
    "count_matches": screw_count == expected_count,
    "predictions": screws,
}, indent=2))
