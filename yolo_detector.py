"""
Optional YOLOv8 person detector.

Ye module person ko detect karke uske bounding box ke around frame ko
crop kar deta hai - isse:
  - Background me chalte-firte doosre log / movement ignore ho jaata hai
  - Multi-person environment (jaise gym) me sirf exercise karne wale
    insaan par pose estimation focus hota hai, jisse accuracy badhti hai

Agar 'ultralytics' package installed NAHI hai, to ye module gracefully
disable ho jata hai aur app bina kisi crash ke normal (full-frame) mode
me chalta rehta hai. Install karne ke liye: pip install ultralytics
"""

import numpy as np

try:
    from ultralytics import YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


class YOLOPersonDetector:
    PERSON_CLASS_ID = 0  # COCO dataset me 'person' class id 0 hoti hai

    def __init__(self, model_name="yolov8n.pt", confidence=0.5, padding_ratio=0.15):
        self.available = False
        self.confidence = confidence
        self.padding_ratio = padding_ratio
        self.model = None
        self.load_error = None

        if not _ULTRALYTICS_AVAILABLE:
            self.load_error = "ultralytics package installed nahi hai (optional feature)"
            return

        try:
            self.model = YOLO(model_name)
            self.available = True
        except Exception as e:  # model load/download fail hone par bhi app crash na ho
            self.load_error = str(e)
            self.available = False

    def detect_person_crop(self, frame_bgr):
        """
        Sabse badi/confident 'person' detection ka bounding box crop karke
        return karta hai. Agar koi person na mile ya detector available na
        ho, to original frame hi wapas kar deta hai (safe fallback).

        Return: (cropped_frame, box_or_None)
        """
        if not self.available:
            return frame_bgr, None

        h, w = frame_bgr.shape[:2]
        try:
            results = self.model.predict(
                frame_bgr, classes=[self.PERSON_CLASS_ID], conf=self.confidence, verbose=False
            )
        except Exception:
            return frame_bgr, None

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return frame_bgr, None

        boxes = results[0].boxes
        areas = []
        for b in boxes.xyxy:
            x1, y1, x2, y2 = b.tolist()
            areas.append((x2 - x1) * (y2 - y1))
        best_idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

        pad_x = (x2 - x1) * self.padding_ratio
        pad_y = (y2 - y1) * self.padding_ratio
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return frame_bgr, None

        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)
