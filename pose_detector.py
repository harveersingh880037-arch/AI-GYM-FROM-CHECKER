import cv2
import mediapipe as mp


class PoseDetector:
    """
    MediaPipe Pose ke upar ek clean wrapper.
    Har frame me body ke 33 landmarks detect karta hai (accuracy ke liye
    model_complexity=1 aur confidence thresholds tuned hai).
    """

    def __init__(self, min_detection_confidence=0.65, min_tracking_confidence=0.65, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        return results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 220, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2
                ),
            )
        return frame

    @staticmethod
    def get_landmark_xy(landmarks, idx, frame_shape):
        """
        Landmark index ke liye pixel coordinates [x, y] aur uski visibility
        (0-1, jitni zyada utna reliable) return karta hai.
        """
        h, w = frame_shape[:2]
        lm = landmarks[idx]
        return [lm.x * w, lm.y * h], lm.visibility
