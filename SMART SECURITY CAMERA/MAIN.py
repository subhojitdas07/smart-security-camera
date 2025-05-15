import cv2
import time
import threading
import requests
import pygame
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ======================= CONFIG ==========================
BOT_TOKEN = 'your bot token here'
CHAT_ID = 'here chat id here'

YOLO_CFG = 'yolov4.cfg'
YOLO_WEIGHTS = 'yolov4.weights'
COCO_NAMES = 'coco.names'
ALARM_SOUND_PATH = 'alarm_sound.mp3'
# ==========================================================

# Init sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)

# Load YOLO
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure image folder exists
if not os.path.exists('detected_images'):
    os.makedirs('detected_images')

# Telegram Functions
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    requests.post(url, data={'chat_id': CHAT_ID, 'text': message})

def send_telegram_image(image_path):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    with open(image_path, 'rb') as image_file:
        files = {'photo': image_file}
        data = {'chat_id': CHAT_ID}
        requests.post(url, files=files, data=data)

# Object Detection Function
def process_frame_for_classification(frame):
    height, width = frame.shape[:2]
    frame_resized = cv2.resize(frame, (416, 416))
    blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(det[0] * width)
                center_y = int(det[1] * height)
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indices is not None and len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Motion Detection Function
fgbg = cv2.createBackgroundSubtractorMOG2()

def process_frame_for_motion(frame, last_motion_time):
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            motion_detected = True

    if motion_detected and time.time() - last_motion_time > 3:
        detected_frame = process_frame_for_classification(frame.copy())
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        image_path = f"detected_images/motion_{timestamp}.jpg"
        cv2.imwrite(image_path, detected_frame)
        threading.Thread(target=send_telegram_message, args=("⚠️ Motion Detected!",)).start()
        threading.Thread(target=send_telegram_image, args=(image_path,)).start()
        threading.Thread(target=alarm_sound.play).start()
        last_motion_time = time.time()

    return frame, last_motion_time

# GUI Live Feed
def live_feed(cap, panel, status_label):
    def update():
        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Camera not detected")
            return
        frame = cv2.resize(frame, (640, 480))
        frame = process_frame_for_classification(frame)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img)
        panel.img_tk = img_tk
        panel.config(image=img_tk)
        status_label.config(text="Live Object Feed Running")
        panel.after(50, update)
    update()

# GUI Motion Feed
def detection_feed(cap, panel, status_label):
    last_motion_time = time.time()
    def update():
        nonlocal last_motion_time
        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Camera not detected")
            return
        frame = cv2.resize(frame, (640, 480))
        frame, last_motion_time = process_frame_for_motion(frame, last_motion_time)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img)
        panel.img_tk = img_tk
        panel.config(image=img_tk)
        status_label.config(text="Motion Detection Feed Running")
        panel.after(50, update)
    update()

# MAIN
cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("SMART SECURITY CAMERA SYSTEM")

frame1 = ttk.Frame(root, padding=10)
frame1.grid(row=0, column=0)
label1 = ttk.Label(frame1)
label1.grid(row=0, column=0)
status1 = ttk.Label(frame1, text="Status: -")
status1.grid(row=1, column=0)

frame2 = ttk.Frame(root, padding=10)
frame2.grid(row=0, column=1)
label2 = ttk.Label(frame2)
label2.grid(row=0, column=0)
status2 = ttk.Label(frame2, text="Status: -")
status2.grid(row=1, column=0)

live_feed(cap, label1, status1)
detection_feed(cap, label2, status2)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
