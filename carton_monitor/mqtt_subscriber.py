import ssl
import paho.mqtt.client as mqtt

HOST = "broker.hivemq.com"
PORT = 8883
TOPIC = "roboflow-demo/tim-4ijf0/carton-inspection/damage"

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected; subscribing to", TOPIC, flush=True)
        client.subscribe(TOPIC, qos=1)
    else:
        print("Connection rejected:", reason_code, flush=True)

def on_subscribe(client, userdata, mid, reason_codes, properties):
    print("Subscribed — waiting for new messages", flush=True)

def on_message(client, userdata, message):
    print("RECEIVED:", message.payload.decode("utf-8", errors="replace"), flush=True)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message

try:
    print(f"Connecting to {HOST}:{PORT}…", flush=True)
    client.connect(HOST, PORT, keepalive=60)
    client.loop_forever()
except Exception as exc:
    print(f"Connection failed: {type(exc).__name__}: {exc}", flush=True)
