import websocket
import json
import time

def on_message(ws, message):
    msg = json.loads(message)
    if msg['op'] == 'publish':
        if msg['topic'] == 'sky360/frames/annotated/compressed':
            data = msg['msg']['data'][:10]  # Truncate the data to first 100 characters
            print(f"Received message from topic {msg['topic']}: header: {header}, data: {data}...")
        elif msg['topic'] == 'sky360/frames/overlayed':
            header = msg['msg']['header']
            data = msg['msg']['data'][:10]  # Truncate the data to first 100 characters
            print(f"Received classified message from topic {msg['topic']}: header: {header}, data: {data}...")

def on_open(ws):
    # Subscribe to annotated topic
    subscribe_msg_annotated = json.dumps({"op": "subscribe",
                                "topic": "sky360/frames/annotated/compressed",
                                "type": "sensor_msgs/msg/CompressedImage"})
    ws.send(subscribe_msg_annotated)

    # Subscribe to overlayed topic
    subscribe_msg_overlayed = json.dumps({"op": "subscribe",
                                "topic": "sky360/frames/overlayed",
                                "type": "sensor_msgs/msg/Image"})
    ws.send(subscribe_msg_overlayed)


# Replace 'docker_container_ip' with the IP address of your Docker container
ws = websocket.WebSocketApp("ws://localhost:8081",
                            on_message=on_message,
                            on_open=on_open)

# Keep the connection alive
ws.run_forever()
