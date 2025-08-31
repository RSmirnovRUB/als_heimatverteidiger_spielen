# -*- coding: utf-8 -*-
"""
Distant Viewing für Videos mit Objekterkennung, Farb- und Helligkeitsanalyse,
Gesichts- und Emotionserkennung, segmentiert in 600 Sekunden HUNK_LENGTH
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from ultralytics import YOLO
from deepface import DeepFace
import os

# === Konfiguration ===
VIDEO_PATH = "/Users/rsmirnov/Desktop/Heimatverteidiger/Sniper Elite VR/10_Sniper_Elite_VR.mp4"  # Pfad zum Video
OUTPUT_DIR = "/Users/rsmirnov/Desktop/dv_output"  # Ausgabeordner
CHUNK_LENGTH = 600  # Segmentlänge in Sekunden

# === Verzeichnisse erstellen ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === YOLOv8 Modell laden ===
model = YOLO("yolov8n.pt")

# === CSV für Ergebnisse ===
results = []

# === Video laden ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

num_chunks = int(duration_sec // CHUNK_LENGTH) + 1

# === Video verarbeiten in Chunks ===
for chunk_idx in range(num_chunks):
    start_frame = int(chunk_idx * CHUNK_LENGTH * fps)
    end_frame = int(min((chunk_idx + 1) * CHUNK_LENGTH * fps, total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    if not frames:
        continue
    
    # === Helligkeit und durchschnittliche Farbe ===
    avg_brightness = sum(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames) / len(frames)
    avg_color = sum(f.mean(axis=(0,1)) for f in frames) / len(frames)  # BGR
    
    # === Objekterkennung mit YOLOv8 ===
    all_objects = []
    for f in frames[::int(fps)]:  # 1 Frame pro Sekunde für YOLO
        results_yolo = model(f)
        for det in results_yolo[0].boxes.cls:
            all_objects.append(model.names[int(det)])
    
    top_objects = Counter(all_objects).most_common(10)
    
    # === Gesichter & Emotionen ===
    face_count = 0
    emotions_detected = Counter()
    for f in frames[::int(fps)]:  # 1 Frame pro Sekunde
        try:
            faces = DeepFace.analyze(f, actions=['emotion'], enforce_detection=False)
            if isinstance(faces, list):
                for face in faces:
                    face_count += 1
                    dominant_emotion = face['dominant_emotion']
                    emotions_detected[dominant_emotion] += 1
            else:
                face_count += 1
                dominant_emotion = faces['dominant_emotion']
                emotions_detected[dominant_emotion] += 1
        except:
            continue
    
    # === Ergebnisse sammeln ===
    results.append({
        "chunk_idx": chunk_idx,
        "avg_brightness": avg_brightness,
        "avg_B": avg_color[0],
        "avg_G": avg_color[1],
        "avg_R": avg_color[2],
        "face_count": face_count,
        "top_objects": dict(top_objects),
        "emotions": dict(emotions_detected)
    })

# === CSV speichern ===
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "dv_results.csv")
df.to_csv(csv_path, index=False)

# === Visualisierung Helligkeit ===
plt.figure(figsize=(6,4))
sns.histplot(df["avg_brightness"], bins=20, kde=True)
plt.title("Helligkeitsverteilung")
plt.xlabel("Helligkeit")
plt.ylabel("Anzahl Segmente")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "brightness_hist.png"))

# === Visualisierung Durchschnittsfarbe ===
avg_color_total = df[["avg_R","avg_G","avg_B"]].mean().astype(int)
color_hex = "#{:02x}{:02x}{:02x}".format(*avg_color_total)

plt.figure(figsize=(2,2))
plt.imshow([[avg_color_total/255]])
plt.axis("off")
plt.title(f"Durchschnittsfarbe {color_hex}")
plt.savefig(os.path.join(OUTPUT_DIR, "avg_color.png"))

# === Top Objekte plotten ===
flat_objects = Counter()
for d in df["top_objects"]:
    flat_objects.update(d)
top10 = flat_objects.most_common(10)
objects, counts = zip(*top10)

plt.figure(figsize=(6,4))
sns.barplot(x=list(counts), y=list(objects), orient="h")
plt.title("Top 10 erkannte Objekte")
plt.xlabel("Anzahl")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_objects.png"))

print("Analyse abgeschlossen! Ergebnisse im Ordner:", OUTPUT_DIR)
