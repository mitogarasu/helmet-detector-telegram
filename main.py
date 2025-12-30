import cv2
import torch
import numpy as np
import time
import requests
from datetime import datetime
import os

# --- TELEGRAM ---
BOT_TOKEN = "ВАШ ТОКЕН"
CHAT_ID = "ВАШ ЧАТ ID"   # строкой, например "123456789" или "-1001234567890"
TG_COOLDOWN_SEC = 10           # раз в 10 секунд максимум
SAVE_DIR = "alerts"
os.makedirs(SAVE_DIR, exist_ok=True)

last_tg_time = 0.0

def tg_send_alert(image_bgr, text: str):
    global last_tg_time
    now = time.time()
    if now - last_tg_time < TG_COOLDOWN_SEC:
        return
    last_tg_time = now

    # сохраняем кадр во временный файл
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(SAVE_DIR, f"no_helmet_{ts}.jpg")
    cv2.imwrite(img_path, image_bgr)

    # отправляем фото + подпись
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(img_path, "rb") as f:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": text},
            files={"photo": f},
            timeout=10
        )
    # если хочешь видеть ошибки:
    if not r.ok:
        print("TG ERROR:", r.status_code, r.text)

# НАСТРОЙКИ
MODEL_PATH = 'models/best.pt'
VIDEO_PATH = ''

CONF_THRES = 0.50         # уверенность детекта
HELMET_IOU_THRES = 0.05   # порог пересечения каски с человеком
NOHELMET_IOU_THRES = 0.10 # если у тебя есть отдельный класс no_helmet (опционально)
HEAD_ZONE = 0.3          # верхняя часть человека (0.45 = верхние 45%)

# ЗАГРУЗКА МОДЕЛИ
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
model.conf = CONF_THRES

print("CLASSES:", model.names)

# ВСПОМОГАТЕЛЬНЫЕ
def iou_xyxy(a, b):
    # a,b: (x1,y1,x2,y2)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def draw_box(frame, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

matched_helmets = []

# ВИДЕО
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

fr = 0

while cap.isOpened():
    time.sleep(0.03)
    ret, frame = cap.read()
    if not ret:
        break

    matched_helmets = []  # ✅ очищаем каждый кадр

    preds = model(frame)
    det = preds.xyxy[0].cpu().numpy()

    persons, helmets, no_helmets = [], [], []

    for x1, y1, x2, y2, conf, cls in det:
        cls = int(cls)
        name = str(model.names[cls]).lower()
        box = (float(x1), float(y1), float(x2), float(y2))
        conf = float(conf)

        if name == "person":
            persons.append((box, conf))
        elif name == "helmet":
            helmets.append((box, conf))
        elif name in ["no helmet", "no_helmet", "without_helmet", "nohelmet"]:
            no_helmets.append((box, conf))

    for p_box, p_conf in persons:
        px1, py1, px2, py2 = p_box
        p_h = py2 - py1

        has_helmet = False

        if no_helmets:
            for nh_box, nh_conf in no_helmets:
                if iou_xyxy(p_box, nh_box) > NOHELMET_IOU_THRES:
                    has_helmet = False
                    break
            else:
                for h_box, h_conf in helmets:
                    if iou_xyxy(p_box, h_box) > HELMET_IOU_THRES:
                        hx1, hy1, hx2, hy2 = h_box
                        h_center_y = (hy1 + hy2) / 2
                        if h_center_y < py1 + HEAD_ZONE * p_h:
                            has_helmet = True
                            matched_helmets.append((h_box, h_conf))  # ✅
                            break
        else:
            for h_box, h_conf in helmets:
                if iou_xyxy(p_box, h_box) > HELMET_IOU_THRES:
                    hx1, hy1, hx2, hy2 = h_box
                    h_center_y = (hy1 + hy2) / 2
                    if h_center_y < py1 + HEAD_ZONE * p_h:
                        has_helmet = True
                        matched_helmets.append((h_box, h_conf))  # ✅ тоже добавили
                        break

        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        label = f"person {p_conf:.2f} {'helmet' if has_helmet else 'NO helmet'}"
        draw_box(frame, p_box, color, label=label)

        if not has_helmet:
            x1, y1, x2, y2 = map(int, p_box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2].copy()

            tg_send_alert(
                frame,
                text=f"🚨 НЕТ ШЛЕМА! 🚨| conf={p_conf:.2f} | frame={fr}"
            )

    # рисуем только каски, которые реально привязались к человеку
    for h_box, h_conf in matched_helmets:
        draw_box(frame, h_box, (0, 255, 0), label=f"helmet {h_conf:.2f}", thickness=2)

    # вывод в консоль
    if fr % 10 == 0:
        print(f"Frame {fr}: persons={len(persons)} helmets={len(helmets)} no_helmets={len(no_helmets)}")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    fr += 1

cap.release()
cv2.destroyAllWindows()