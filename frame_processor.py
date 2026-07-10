"""
Webcam (live) aur uploaded video - dono modes isi ek shared pipeline ko
use karte hai. Isse code duplicate nahi hota aur dono jagah bilkul same
accuracy/logic milti hai (production-ready design principle: single
source of truth for the core analysis logic).
"""

import cv2


def draw_overlay(frame, reps, form_score, feedback, exercise):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, f"{exercise}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Reps: {reps}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Form Score: {form_score}%", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    color = (0, 0, 255) if "GALTI" in feedback else (0, 255, 0)
    cv2.putText(frame, feedback, (230, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    return frame


def draw_angle_marker(frame, point, angle):
    x, y = int(point[0]), int(point[1])
    color = (255, 200, 0)
    cv2.circle(frame, (x, y), 8, color, -1)
    cv2.putText(frame, f"{int(angle)} deg", (x + 12, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def process_frame(frame_bgr, exercise, pose_detector, squat_analyzer, curl_analyzer,
                   mp_pose, yolo_detector=None, use_yolo=False):
    """
    Ek single frame process karta hai:
      1) (optional) YOLO se person crop - background/multi-person noise filter
      2) MediaPipe se pose landmarks detect
      3) Exercise-specific analyzer se angle/reps/feedback nikaalna
      4) Frame par skeleton + angle visualization + stats overlay draw karna

    Return: (annotated_frame, state_dict)
    """
    working_frame = frame_bgr
    used_yolo_box = None

    if use_yolo and yolo_detector is not None and yolo_detector.available:
        cropped, box = yolo_detector.detect_person_crop(frame_bgr)
        if box is not None:
            working_frame = cropped
            used_yolo_box = box

    results = pose_detector.process(working_frame)
    annotated = working_frame.copy()

    # Default state ACTIVE analyzer ke persisted counters se banta hai -
    # isse agar kisi frame me landmark detect na ho (jaise thodi der ke
    # liye person frame se bahar chala gaya), to reps counter galti se
    # 0 par reset nahi hota, current count hi dikhta rehta hai.
    active_analyzer = squat_analyzer if exercise == "Squat" else curl_analyzer
    state = {
        "feedback": "Sharir camera me detect nahi ho raha",
        "reps": active_analyzer.counter,
        "good_reps": active_analyzer.good_reps,
        "bad_reps": active_analyzer.bad_reps,
        "form_score": active_analyzer.form_score,
        "angle": None,
        "joint_point": None,
        "last_rep_event": None,
    }

    if results.pose_landmarks:
        annotated = pose_detector.draw_landmarks(annotated, results)
        landmarks = results.pose_landmarks.landmark
        try:
            if exercise == "Squat":
                state = squat_analyzer.analyze(landmarks, annotated.shape, mp_pose)
            else:
                state = curl_analyzer.analyze(landmarks, annotated.shape, mp_pose)
        except Exception:
            state["feedback"] = "Position thik se dikhaao"

        if state.get("angle") is not None and state.get("joint_point") is not None:
            annotated = draw_angle_marker(annotated, state["joint_point"], state["angle"])

    annotated = draw_overlay(annotated, state["reps"], state["form_score"], state["feedback"], exercise)
    state["used_yolo_box"] = used_yolo_box
    return annotated, state
