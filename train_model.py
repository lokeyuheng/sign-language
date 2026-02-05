from training.train_lstm import train_lstm
import json
import os

# Make sure inference folder exists
os.makedirs("inference", exist_ok=True)

# 👇 PUT YOUR OWN TEST VIDEOS HERE
video_paths = [
    "test_videos/hello.mp4",
    "test_videos/thanks.mp4"
]

# 👇 numeric labels
labels = [0, 1]

label_map = {
    0: "hello",
    1: "thanks"
}

# Save label map
with open("inference/label_map.json", "w") as f:
    json.dump(label_map, f)

train_lstm(
    video_paths=video_paths,
    labels=labels,
    num_classes=len(label_map)
)
